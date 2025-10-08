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
        use_kan_flow: bool = False
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
            dropout=dropout
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
            'feat_8_0': self.downsample(feat_match_0).flatten(2, 3),
            'feat_8_1': self.downsample(feat_match_1).flatten(2, 3),
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
        
        # Step 3: Local Path - Process with adaptive spans
        local_match_0, local_match_1, local_geom_0, local_geom_1 = self.local_mamba(
            feat_match_0, feat_match_1,
            feat_geom_0, feat_geom_1,
            adaptive_spans_x[:data['bs']], adaptive_spans_y[:data['bs']],
            adaptive_spans_x[data['bs']:], adaptive_spans_y[data['bs']:]
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


class LocalAdaptiveMamba(nn.Module):
    """
    Local Mamba processing with adaptive spans.
    
    Processes features within adaptive local windows determined by flow predictions.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        depth: int = 4,
        d_geom: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_geom = d_geom
        
        # Local Mamba blocks
        from .mamba_module import create_multihead_block
        self.mamba_blocks = nn.ModuleList([
            create_multihead_block(
                d_model=feature_dim,
                d_geom=d_geom,
                rms_norm=True,
                residual_in_fp32=True,
                layer_idx=i
            )
            for i in range(depth)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        feat_match_0: torch.Tensor,
        feat_match_1: torch.Tensor,
        feat_geom_0: torch.Tensor,
        feat_geom_1: torch.Tensor,
        spans_x_0: torch.Tensor,
        spans_y_0: torch.Tensor,
        spans_x_1: torch.Tensor,
        spans_y_1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process features with adaptive local spans.
        
        Args:
            feat_match_0/1: Matching features (B, C, H, W)
            feat_geom_0/1: Geometric features (B, C_geom, H, W)
            spans_x/y_0/1: Adaptive spans for each position (B, H, W)
            
        Returns:
            Updated matching and geometric features
        """
        B, C, H, W = feat_match_0.shape
        
        # For simplicity, use average span for the entire feature map
        # In full implementation, this would process each position with its specific span
        avg_span = int(torch.mean(torch.cat([spans_x_0, spans_y_0]).float()).item())
        avg_span = max(3, min(avg_span, 15))  # Clamp to reasonable range
        
        # Extract local windows with average span
        # This is a simplified version - full implementation would use position-specific spans
        padding = avg_span // 2
        feat_match_0_padded = F.pad(feat_match_0, (padding, padding, padding, padding))
        feat_match_1_padded = F.pad(feat_match_1, (padding, padding, padding, padding))
        
        # Unfold to get local windows
        windows_0 = F.unfold(feat_match_0_padded, kernel_size=avg_span, stride=1)
        windows_1 = F.unfold(feat_match_1_padded, kernel_size=avg_span, stride=1)
        
        # Reshape for processing
        windows_0 = rearrange(windows_0, 'b (c k) (h w) -> b (h w) (k c)', 
                            c=C, h=H, w=W, k=avg_span**2)
        windows_1 = rearrange(windows_1, 'b (c k) (h w) -> b (h w) (k c)', 
                            c=C, h=H, w=W, k=avg_span**2)
        
        # Process through Mamba blocks
        for block in self.mamba_blocks:
            windows_0_match, windows_0_geom = block(windows_0)
            windows_1_match, windows_1_geom = block(windows_1)
            windows_0 = self.dropout(windows_0_match)
            windows_1 = self.dropout(windows_1_match)
        
        # Aggregate back to spatial dimensions
        # Simple average pooling over windows
        processed_0 = windows_0_match.mean(dim=2).view(B, H, W, C).permute(0, 3, 1, 2)
        processed_1 = windows_1_match.mean(dim=2).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # For geometric features
        processed_geom_0 = windows_0_geom.mean(dim=2).view(B, H, W, -1).permute(0, 3, 1, 2)
        processed_geom_1 = windows_1_geom.mean(dim=2).view(B, H, W, -1).permute(0, 3, 1, 2)
        
        return processed_0, processed_1, processed_geom_0, processed_geom_1


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