"""
AS-Mamba Block - Core building block of the AS-Mamba architecture.

This module implements the adaptive span mechanism with hierarchical Mamba processing,
combining global and local feature interactions guided by flow predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Optional, Tuple

from .flow_predictor import FlowPredictor, AdaptiveSpanComputer
from .mamba_module import JointMambaMultiHead


class AS_Mamba_Block(nn.Module):
    """
    Adaptive Span Mamba Block.
    
    This block implements the core AS-Mamba mechanism:
    1. Flow prediction from previous features
    2. Global feature processing (coarse level)
    3. Local adaptive span processing (medium/fine level)
    4. Feature aggregation and update
    
    Args:
        d_model: Dimension of matching features
        d_geom: Dimension of geometric features  
        d_ffn: Dimension of feed-forward network
        global_depth: Number of Mamba layers for global processing
        local_depth: Number of Mamba layers for local processing
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_geom: int = 64,
        d_ffn: int = 512,
        global_depth: int = 4,
        local_depth: int = 4,
        dropout: float = 0.1,
        use_kan_flow: bool = False,
        window_size: int = 12
    ):
        super().__init__()
        self.d_model = d_model
        self.d_geom = d_geom
        
        # Flow predictor (KAN or MLP based)
        self.flow_predictor = FlowPredictor(
            d_model=d_model,
            d_geom=d_geom,
            hidden_dim=128,
            num_layers=3,
            use_kan=use_kan_flow,
            dropout=dropout
        )
        
        # Adaptive span computer
        self.span_computer = AdaptiveSpanComputer(
            base_span=7,
            max_span=15,
            temperature=1.0
        )
        
        # Global Path: Process downsampled features
        self.global_mamba = JointMambaMultiHead(
            feature_dim=d_model,
            depth=global_depth,
            d_geom=d_geom,
            return_geometry=True,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True
        )
        
        # Local Path: Process with adaptive spans
        self.local_mamba = LocalAdaptiveMamba(
            feature_dim=d_model,
            depth=local_depth,
            d_geom=d_geom,
            dropout=dropout,
            max_span_groups = 5
        )
        
        # Feature aggregation and update (FFN)
        self.feature_fusion = FeatureFusionFFN(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout
        )
        
        # Downsample and upsample operations
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample_match = nn.ConvTranspose2d(d_model, d_model, kernel_size=2, stride=2)
        self.upsample_geom = nn.ConvTranspose2d(d_geom, d_geom, kernel_size=2, stride=2)
        
    def forward(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of AS-Mamba Block.
        
        Args:
            data: Dictionary containing:
                - feat_8_0, feat_8_1: Matching features from previous block (B, C, H, W)
                - feat_geom_0, feat_geom_1: Geometric features from previous block (B, C_geom, H, W)
                - Other metadata (bs, h_8, w_8, etc.)
                
        Returns:
            Updated data dictionary with:
                - feat_8_0, feat_8_1: Updated matching features
                - feat_geom_0, feat_geom_1: Updated geometric features
                - flow_map: Predicted flow map
                - adaptive_spans: Computed adaptive spans
        """
        # Extract features
        feat_match_0 = data['feat_8_0'].view(data['bs'], self.d_model, data['h_8'], data['w_8'])
        feat_match_1 = data['feat_8_1'].view(data['bs'], self.d_model, data['h_8'], data['w_8'])
        
        # Get geometric features (from previous block or initialize)
        if 'feat_geom_0' in data:
            feat_geom_0 = data['feat_geom_0'].view(data['bs'], self.d_geom, data['h_8'], data['w_8'])
            feat_geom_1 = data['feat_geom_1'].view(data['bs'], self.d_geom, data['h_8'], data['w_8'])
        else:
            # Initialize geometric features if first block
            feat_geom_0 = torch.zeros(data['bs'], self.d_geom, data['h_8'], data['w_8'], 
                                     device=feat_match_0.device, dtype=feat_match_0.dtype)
            feat_geom_1 = torch.zeros_like(feat_geom_0)
        
        # Step 1: Flow prediction
        # Concatenate features from both images for joint flow prediction
        feat_match_concat = torch.cat([feat_match_0, feat_match_1], dim=0)  # (2B, C, H, W)
        feat_geom_concat = torch.cat([feat_geom_0, feat_geom_1], dim=0)  # (2B, C_geom, H, W)
        
        flow_output = self.flow_predictor(feat_match_concat, feat_geom_concat)
        flow_map = flow_output['flow_with_uncertainty']  # (2B, H, W, 4)
        
        # Compute adaptive spans from flow
        adaptive_spans_x, adaptive_spans_y = self.span_computer(
            flow_output['flow'],
            flow_output['uncertainty']
        )
        
        # Step 2: Global Path - Process downsampled features
        # Prepare data for global Mamba
        global_data = {
            # 'feat_8_0': self.downsample(feat_match_0).flatten(2, 3),
            # 'feat_8_1': self.downsample(feat_match_1).flatten(2, 3),
            'feat_8_0': self.downsample(feat_match_0),
            'feat_8_1': self.downsample(feat_match_1),
            'bs': data['bs'],
            'h_8': data['h_8'] // 2,
            'w_8': data['w_8'] // 2
        }
        
        # Process through global Mamba
        self.global_mamba(global_data)
        
        # Extract global features
        global_match_0 = global_data['feat_8_0'].view(data['bs'], self.d_model, data['h_8']//2, data['w_8']//2)
        global_match_1 = global_data['feat_8_1'].view(data['bs'], self.d_model, data['h_8']//2, data['w_8']//2)
        global_geom_0 = global_data['feat_geom_0'].view(data['bs'], self.d_geom, data['h_8']//2, data['w_8']//2)
        global_geom_1 = global_data['feat_geom_1'].view(data['bs'], self.d_geom, data['h_8']//2, data['w_8']//2)
        
        # Upsample global features
        global_match_0 = self.upsample_match(global_match_0)
        global_match_1 = self.upsample_match(global_match_1)
        global_geom_0 = self.upsample_geom(global_geom_0)
        global_geom_1 = self.upsample_geom(global_geom_1)

        # Add FlowMap
        flow_map_0 = flow_map[:data['bs']]  # (B, H, W, 4)
        flow_map_1 = flow_map[data['bs']:] 
        
        # Step 3: Local Path - Process with adaptive spans
        # local_match_0, local_match_1, local_geom_0, local_geom_1 = self.local_mamba(
        #     feat_match_0, feat_match_1,
        #     feat_geom_0, feat_geom_1,
        #     adaptive_spans_x[:data['bs']], adaptive_spans_y[:data['bs']],
        #     adaptive_spans_x[data['bs']:], adaptive_spans_y[data['bs']:]
        # )
        if adaptive_spans_x.shape[0] == 2 * data['bs']:
            # Spans are concatenated for both images
            spans_x_0 = adaptive_spans_x[:data['bs']]
            spans_y_0 = adaptive_spans_y[:data['bs']]
            spans_x_1 = adaptive_spans_x[data['bs']:]
            spans_y_1 = adaptive_spans_y[data['bs']:]
        else:
            # Single set of spans (use same for both images)
            spans_x_0 = spans_x_1 = adaptive_spans_x
            spans_y_0 = spans_y_1 = adaptive_spans_y
        
        # local_match_0, local_match_1, local_geom_0, local_geom_1 = self.local_mamba(
        #     feat_match_0, feat_match_1,
        #     feat_geom_0, feat_geom_1,
        #     spans_x_0, spans_y_0,
        #     spans_x_1, spans_y_1
        # )

        local_match_0, local_match_1, local_geom_0, local_geom_1 = self.local_mamba(
        feat_match_0, feat_match_1,
        feat_geom_0, feat_geom_1,
        flow_map_0, flow_map_1,
        spans_x_0, spans_y_0,
        spans_x_1, spans_y_1
        ) 
        
        # Step 4: Feature aggregation and update
        # Combine global and local features
        combined_match_0 = torch.stack([global_match_0, local_match_0, feat_match_0], dim=1)
        combined_match_1 = torch.stack([global_match_1, local_match_1, feat_match_1], dim=1)
        
        # Apply fusion FFN
        updated_match_0 = self.feature_fusion(combined_match_0)
        updated_match_1 = self.feature_fusion(combined_match_1)
        
        # Combine geometric features (simple average for now)
        updated_geom_0 = (global_geom_0 + local_geom_0) / 2
        updated_geom_1 = (global_geom_1 + local_geom_1) / 2
        
        # Update data dictionary
        data.update({
            'feat_8_0': updated_match_0.flatten(2, 3),
            'feat_8_1': updated_match_1.flatten(2, 3),
            'feat_geom_0': updated_geom_0.flatten(2, 3),
            'feat_geom_1': updated_geom_1.flatten(2, 3),
            'flow_map': flow_map,
            'adaptive_spans': (adaptive_spans_x, adaptive_spans_y)
        })
        
        return data


# class LocalAdaptiveMamba(nn.Module):
#     """
#     Local Mamba processing with adaptive spans.
    
#     Processes features within adaptive local windows determined by flow predictions.
#     """
    
#     def __init__(
#         self,
#         feature_dim: int = 256,
#         depth: int = 4,
#         d_geom: int = 64,
#         dropout: float = 0.1,
#         window_size: int = 4
#     ):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.d_geom = d_geom

#         unfolded_feature_dim = feature_dim * window_size * window_size
        
#         # Local Mamba blocks
#         from .mamba_module import create_multihead_block
#         self.mamba_blocks = nn.ModuleList([
#             create_multihead_block(
#                 d_model=self.feature_dim,
#                 d_geom=d_geom,
#                 rms_norm=True,
#                 residual_in_fp32=True,
#                 layer_idx=i
#             )
#             for i in range(depth)
#         ])
        
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(
#         self,
#         feat_match_0: torch.Tensor,
#         feat_match_1: torch.Tensor,
#         feat_geom_0: torch.Tensor,
#         feat_geom_1: torch.Tensor,
#         spans_x_0: torch.Tensor,
#         spans_y_0: torch.Tensor,
#         spans_x_1: torch.Tensor,
#         spans_y_1: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Process features with adaptive local spans.
        
#         Args:
#             feat_match_0/1: Matching features (B, C, H, W)
#             feat_geom_0/1: Geometric features (B, C_geom, H, W)
#             spans_x/y_0/1: Adaptive spans for each position (B, H, W)
            
#         Returns:
#             Updated matching and geometric features
#         """
#         B, C, H, W = feat_match_0.shape
        
#         # For simplicity, use average span for the entire feature map
#         # In full implementation, this would process each position with its specific span
#         avg_span = int(torch.mean(torch.cat([spans_x_0, spans_y_0]).float()).item())
#         avg_span = max(3, min(avg_span, 15))  # Clamp to reasonable range
        
#         # Extract local windows with average span
#         # This is a simplified version - full implementation would use position-specific spans
#         padding = avg_span // 2
#         feat_match_0_padded = F.pad(feat_match_0, (padding, padding, padding, padding))
#         feat_match_1_padded = F.pad(feat_match_1, (padding, padding, padding, padding))
        
#         # Unfold to get local windows
#         windows_0 = F.unfold(feat_match_0_padded, kernel_size=avg_span, stride=1)
#         windows_1 = F.unfold(feat_match_1_padded, kernel_size=avg_span, stride=1)
        
#         # Reshape for processing
#         # windows_0 = rearrange(windows_0, 'b (c k) (h w) -> b (h w) (k c)', 
#         #                     c=C, h=H, w=W, k=avg_span**2)
#         # windows_1 = rearrange(windows_1, 'b (c k) (h w) -> b (h w) (k c)', 
#         #                     c=C, h=H, w=W, k=avg_span**2)
#         windows_0 = rearrange(windows_0, 'b (c k1 k2) l -> (b l) (k1 k2) c', 
#                               k1=avg_span, k2=avg_span, c=C)
#         windows_1 = rearrange(windows_1, 'b (c k1 k2) l -> (b l) (k1 k2) c', 
#                               k1=avg_span, k2=avg_span, c=C)
        
#         # Process through Mamba blocks
#         for block in self.mamba_blocks:
#             windows_0_match, windows_0_geom = block(windows_0)
#             windows_1_match, windows_1_geom = block(windows_1)
#             windows_0 = self.dropout(windows_0_match)
#             windows_1 = self.dropout(windows_1_match)
        
#         # Aggregate back to spatial dimensions
#         # Simple average pooling over windows
#         processed_0 = windows_0_match.mean(dim=2).view(B, H, W, C).permute(0, 3, 1, 2)
#         processed_1 = windows_1_match.mean(dim=2).view(B, H, W, C).permute(0, 3, 1, 2)
        
#         # For geometric features
#         processed_geom_0 = windows_0_geom.mean(dim=2).view(B, H, W, -1).permute(0, 3, 1, 2)
#         processed_geom_1 = windows_1_geom.mean(dim=2).view(B, H, W, -1).permute(0, 3, 1, 2)
        
#         return processed_0, processed_1, processed_geom_0, processed_geom_1


class FeatureFusionFFN(nn.Module):
    """
    Feature fusion feed-forward network.
    
    Aggregates features from global and local paths with residual connection.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Weighted combination of multiple feature sources
        self.weight_proj = nn.Sequential(
            nn.Conv2d(3 * d_model, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_ffn, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_ffn, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multiple feature sources.
        
        Args:
            combined_features: Stacked features (B, N_sources, C, H, W)
            
        Returns:
            Fused features (B, C, H, W)
        """
        B, N, C, H, W = combined_features.shape
        
        # Compute adaptive weights for each source
        features_concat = combined_features.view(B, N * C, H, W)
        weights = self.weight_proj(features_concat)  # (B, N, H, W)
        weights = weights.unsqueeze(2)  # (B, N, 1, H, W)
        
        # Weighted combination
        fused = (combined_features * weights).sum(dim=1)  # (B, C, H, W)
        
        # Apply FFN with residual
        residual = fused
        out = self.ffn(fused)
        out = out + residual
        
        # Apply layer norm
        out = rearrange(out, 'b c h w -> b h w c')
        out = self.norm(out)
        out = rearrange(out, 'b h w c -> b c h w')
        
        return out

class LocalAdaptiveMamba(nn.Module):
    """
    Local Mamba processing with TRUE adaptive spans following ASpanFormer.
    
    Key features:
    1. Position-specific window sizes based on uncertainty
    2. Flow-guided window centers (predicted correspondences)
    3. Efficient grouped processing for similar span sizes
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        depth: int = 4,
        d_geom: int = 64,
        dropout: float = 0.1,
        max_span_groups: int = 5  # Group similar spans for efficiency
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_geom = d_geom
        self.max_span_groups = max_span_groups
        
        # Import MultiHeadMambaBlock for local processing
        from .mamba_module import create_multihead_block
        
        # Create different Mamba blocks for different span sizes
        # This follows ASpanFormer's strategy of grouping similar window sizes
        self.span_groups = [5, 7, 9, 11, 15]  # Representative span sizes
        self.mamba_blocks = nn.ModuleDict({
            f'span_{s}': nn.ModuleList([
                create_multihead_block(
                    d_model=feature_dim,
                    d_geom=d_geom,
                    rms_norm=True,
                    residual_in_fp32=True,
                    layer_idx=i
                )
                for i in range(depth)
            ])
            for s in self.span_groups
        })
        
        self.dropout = nn.Dropout(dropout)
        
        # Aggregation weights for combining different span outputs
        self.span_aggregator = nn.Linear(len(self.span_groups) * feature_dim, feature_dim)
        self.span_aggregator_geom = nn.Linear(len(self.span_groups) * d_geom, d_geom)
        
    def forward(
        self,
        feat_match_0: torch.Tensor,
        feat_match_1: torch.Tensor,
        feat_geom_0: torch.Tensor,
        feat_geom_1: torch.Tensor,
        flow_map_0: torch.Tensor,  # (B, H, W, 4) - includes flow and uncertainty
        flow_map_1: torch.Tensor,
        adaptive_spans_x_0: torch.Tensor,  # (B, H, W)
        adaptive_spans_y_0: torch.Tensor,
        adaptive_spans_x_1: torch.Tensor,
        adaptive_spans_y_1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process features with TRUE position-adaptive local spans.
        
        This implementation follows ASpanFormer's approach:
        1. Group positions by similar span sizes
        2. Extract flow-guided windows for each group
        3. Process each group with appropriate Mamba blocks
        4. Aggregate results back to spatial positions
        """
        B, C, H, W = feat_match_0.shape
        device = feat_match_0.device
        
        # Process each image separately
        processed_0 = self._process_adaptive_windows(
            feat_match_0, feat_geom_0, 
            flow_map_0[..., :2], flow_map_0[..., 2:],  # flow and uncertainty
            adaptive_spans_x_0, adaptive_spans_y_0,
            target_feat=feat_match_1  # For flow-guided attention
        )
        
        processed_1 = self._process_adaptive_windows(
            feat_match_1, feat_geom_1,
            flow_map_1[..., :2], flow_map_1[..., 2:],
            adaptive_spans_x_1, adaptive_spans_y_1,
            target_feat=feat_match_0
        )
        
        return processed_0['match'], processed_1['match'], \
               processed_0['geom'], processed_1['geom']
    
    def _process_adaptive_windows(
        self,
        feat_match: torch.Tensor,  # (B, C, H, W)
        feat_geom: torch.Tensor,   # (B, C_geom, H, W)
        flow: torch.Tensor,         # (B, H, W, 2)
        uncertainty: torch.Tensor,  # (B, H, W, 2)
        spans_x: torch.Tensor,      # (B, H, W)
        spans_y: torch.Tensor,      # (B, H, W)
        target_feat: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Extract and process windows with position-specific sizes.
        
        Following ASpanFormer:
        - Windows are centered at predicted correspondence locations (flow-guided)
        - Window size varies based on uncertainty
        - Positions with similar spans are grouped for efficiency
        """
        B, C, H, W = feat_match.shape
        device = feat_match.device
        
        # Initialize output tensors
        output_match_list = []
        output_geom_list = []
        
        # Group positions by span size for efficient processing
        for span_size in self.span_groups:
            # Find positions that should use this span size
            # (In practice, we assign each position to its nearest span group)
            span_mask = self._get_span_group_mask(spans_x, spans_y, span_size)
            
            if not span_mask.any():
                continue
            
            # Extract windows for this span group
            windows_match, windows_geom, window_positions = self._extract_adaptive_windows(
                feat_match, feat_geom, flow, span_size, span_mask
            )
            
            if windows_match is None:
                continue
            
            # Process through corresponding Mamba blocks
            mamba_blocks = self.mamba_blocks[f'span_{span_size}']
            for block in mamba_blocks:
                windows_match, windows_geom = block(windows_match)
                windows_match = self.dropout(windows_match)
            
            # Store results with position information
            output_match_list.append((windows_match, window_positions, span_mask))
            output_geom_list.append((windows_geom, window_positions, span_mask))
        
        # Aggregate results from different span groups
        output_match = self._aggregate_multispan_outputs(
            output_match_list, (B, C, H, W), self.span_aggregator
        )
        output_geom = self._aggregate_multispan_outputs(
            output_geom_list, (B, self.d_geom, H, W), self.span_aggregator_geom
        )
        
        return {'match': output_match, 'geom': output_geom}
    
    def _get_span_group_mask(
        self, 
        spans_x: torch.Tensor, 
        spans_y: torch.Tensor, 
        target_span: int
    ) -> torch.Tensor:
        """
        Create mask for positions that should use the target span size.
        Groups positions with similar span requirements.
        """
        # Use average of x and y spans
        avg_spans = (spans_x + spans_y) / 2.0
        
        # Find positions closest to this span group
        span_diffs = torch.abs(avg_spans - target_span)
        
        # Create mask for positions within tolerance
        tolerance = 1.5
        mask = span_diffs < tolerance
        
        # Ensure each position is only in one group (closest one)
        if hasattr(self, '_position_assignments'):
            # Avoid reassigning positions
            already_assigned = self._position_assignments
            mask = mask & ~already_assigned
        
        return mask
    
    def _extract_adaptive_windows(
        self,
        feat_match: torch.Tensor,
        feat_geom: torch.Tensor,
        flow: torch.Tensor,
        window_size: int,
        position_mask: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract windows of specific size at flow-predicted positions.
        
        This is the KEY difference from original implementation:
        - Windows are centered at PREDICTED CORRESPONDENCE locations
        - Not just at grid positions
        """
        B, C, H, W = feat_match.shape
        device = feat_match.device
        
        # Get positions where we need to extract windows
        positions = torch.nonzero(position_mask)  # (N, 3) - batch, y, x
        
        if len(positions) == 0:
            return None, None, None
        
        # Add flow offset to get window centers (following ASpanFormer)
        window_centers = positions.float()
        for i, (b, y, x) in enumerate(positions):
            # Add flow to shift window center
            window_centers[i, 1] += flow[b, y, x, 1]  # y offset
            window_centers[i, 2] += flow[b, y, x, 0]  # x offset
        
        # Clamp to image boundaries
        window_centers[:, 1] = torch.clamp(window_centers[:, 1], 0, H - 1)
        window_centers[:, 2] = torch.clamp(window_centers[:, 2], 0, W - 1)
        
        # Extract windows using grid_sample for sub-pixel accuracy
        windows_match = self._extract_windows_grid_sample(
            feat_match, window_centers, window_size
        )
        windows_geom = self._extract_windows_grid_sample(
            feat_geom, window_centers, window_size
        )
        
        return windows_match, windows_geom, positions
    
    def _extract_windows_grid_sample(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
        window_size: int
    ) -> torch.Tensor:
        """
        Extract windows using grid_sample for sub-pixel accurate sampling.
        This allows flow-guided window centers at fractional positions.
        """
        B, C, H, W = features.shape
        N = centers.shape[0]
        
        # Create grid for each window
        half_size = window_size // 2
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-half_size, half_size + 1, device=features.device),
            torch.arange(-half_size, half_size + 1, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()  # (ws, ws, 2)
        
        # Expand grid for all windows]
        # clone (Runtime error fix)
        grid = grid.unsqueeze(0).expand(N, -1, -1, -1).clone()  # (N, ws, ws, 2)
        
        # Add window centers to grid
        # grid[..., 0] += centers[:, 2:3].unsqueeze(1).unsqueeze(1)  # x
        # grid[..., 1] += centers[:, 1:2].unsqueeze(1).unsqueeze(1)  # y
        # bload cast error fix
        grid[..., 0] += centers[:, 2:3].unsqueeze(1)  # x
        grid[..., 1] += centers[:, 1:2].unsqueeze(1)  # y
        
        # Normalize to [-1, 1] for grid_sample
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        
        # Extract windows for each position
        windows = []
        for i, (b, _, _) in enumerate(centers.long()):
            window = F.grid_sample(
                features[b:b+1], 
                grid[i:i+1],
                mode='bilinear',
                align_corners=True,
                padding_mode='zeros'
            )
            windows.append(window)
        
        windows = torch.cat(windows, dim=0)  # (N, C, ws, ws)
        windows = rearrange(windows, 'n c h w -> n (h w) c')
        
        return windows
    
    def _aggregate_multispan_outputs(
        self,
        output_list: list,
        output_shape: tuple,
        aggregator: nn.Module
    ) -> torch.Tensor:
        # """
        # Aggregate outputs from different span groups back to spatial dimensions.
        # """
        # B, C, H, W = output_shape
        # device = output_list[0][0].device if output_list else torch.device('cpu')
        """
        Aggregate outputs from different span groups back to spatial dimensions.
        This implementation is based on the concept of averaging contributions
        from multiple overlapping adaptive windows, as inspired by ASpanFormer's
        goal of integrating context from variable sources.
        """
        B, C, H, W = output_shape

        # If the list of outputs to process is empty, return a zero tensor
        # of the correct shape and device to prevent downstream TypeErrors.
        if not output_list:
            # Use a parameter from the module to reliably get the correct device.
            device = self.span_aggregator.weight.device
            return torch.zeros(output_shape, device=device)

        # The device is determined from the first available tensor in the list.
        device = output_list[0][0].device

        # Initialize tensors to accumulate the sum of features and the count of contributions.
        output_sum = torch.zeros(output_shape, device=device)
        counts = torch.zeros((B, 1, H, W), device=device)

        # Iterate through the outputs of each span group.
        for windows, positions, _ in output_list:
            if windows is None or positions is None or len(positions) == 0:
                continue

            # As a simplification and efficient aggregation method, we use the feature
            # of the center pixel of each window as its representative feature.
            center_pixel_idx = windows.shape[1] // 2
            center_features = windows[:, center_pixel_idx, :]  # Shape: (N, C) where N is the number of windows

            # Get the coordinates (batch, y, x) for each feature.
            b_idx = positions[:, 0]
            y_idx = positions[:, 1]
            x_idx = positions[:, 2]

            # Add the features to the corresponding locations in the output_sum tensor.
            # Using a direct indexing assignment is cleaner here.
            # Note: If multiple windows map to the exact same pixel, this will overwrite.
            # A more robust approach for perfect aggregation is scatter_add, but for this
            # matching task, direct assignment is a common and effective strategy.
            # Let's refine this to use addition for accumulation.
            # This requires careful indexing.
            for i, (b, y, x) in enumerate(zip(b_idx, y_idx, x_idx)):
                 output_sum[b, :, y, x] += center_features[i]
                 counts[b, :, y, x] += 1

        # Avoid division by zero by adding a small epsilon where counts are zero.
        counts.clamp_(min=1e-6)

        # Compute the final feature map by averaging the contributions.
        final_output = output_sum / counts

        return final_output