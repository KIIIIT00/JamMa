"""
Flow Predictor module for AS-Mamba.

This module predicts optical flow and uncertainty maps from matching and geometric features,
following the adaptive span mechanism from ASpanFormer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict


class FlowPredictor(nn.Module):
    """
    Flow Predictor for adaptive span computation.
    
    Predicts a flow map Φ that contains:
    - (x, y): predicted displacement/flow vectors
    - (σ_x, σ_y): uncertainty/confidence in each direction
    
    This is used to determine the adaptive span for local attention in AS-Mamba blocks.
    
    Args:
        d_model (int): Dimension of matching features
        d_geom (int): Dimension of geometric features
        hidden_dim (int): Hidden dimension for MLP layers
        num_layers (int): Number of MLP layers
        use_kan (bool): Whether to use KAN (Kolmogorov-Arnold Networks) instead of MLP
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_geom: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        use_kan: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_geom = d_geom
        self.hidden_dim = hidden_dim
        self.use_kan = use_kan
        
        # Input projection - combine matching and geometric features
        input_dim = d_model + d_geom
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Main prediction network
        if use_kan:
            # KAN implementation (simplified version)
            # In full implementation, this would use learnable activation functions
            self.flow_net = self._build_kan_network(hidden_dim, num_layers)
        else:
            # Standard MLP
            self.flow_net = self._build_mlp_network(hidden_dim, num_layers, dropout)
        
        # Output heads for flow and uncertainty
        # Output: (delta_x, delta_y, log_sigma_x, log_sigma_y)
        self.flow_head = nn.Linear(hidden_dim, 4)
        
        # Optional: Separate heads for flow and uncertainty
        self.use_separate_heads = False
        if self.use_separate_heads:
            self.flow_xy_head = nn.Linear(hidden_dim, 2)  # (x, y)
            self.uncertainty_head = nn.Linear(hidden_dim, 2)  # (σ_x, σ_y)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize output layer with small values for stable training
        nn.init.normal_(self.flow_head.weight, std=0.01)
        nn.init.zeros_(self.flow_head.bias)
    
    def _build_mlp_network(self, hidden_dim: int, num_layers: int, dropout: float) -> nn.Module:
        """Build standard MLP network."""
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
        return nn.Sequential(*layers)
    
    def _build_kan_network(self, hidden_dim: int, num_layers: int) -> nn.Module:
        """
        Build KAN (Kolmogorov-Arnold Network) - simplified version.
        
        In a full KAN implementation, this would include:
        - Learnable activation functions (e.g., using splines or polynomials)
        - Adaptive basis functions
        
        For now, we use a simplified version with learnable weighted combinations
        of different activation functions.
        """
        class KANLayer(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = nn.Linear(in_features, out_features * 4)  # 4 different paths
                self.combine = nn.Linear(out_features * 4, out_features)
                self.norm = nn.LayerNorm(out_features)
                
            def forward(self, x):
                # Apply different activation functions
                h = self.linear(x)
                h1, h2, h3, h4 = torch.chunk(h, 4, dim=-1)
                
                # Different activation functions (simplified KAN)
                h1 = torch.sin(h1)  # Periodic activation
                h2 = torch.relu(h2)  # ReLU
                h3 = torch.tanh(h3)  # Tanh
                h4 = h4 * torch.sigmoid(h4)  # Swish/SiLU
                
                # Combine with learnable weights
                h = torch.cat([h1, h2, h3, h4], dim=-1)
                h = self.combine(h)
                return self.norm(h)
        
        layers = []
        for _ in range(num_layers):
            layers.append(KANLayer(hidden_dim, hidden_dim))
        return nn.Sequential(*layers)
    
    def forward(
        self,
        feat_match: torch.Tensor,
        feat_geom: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict flow and uncertainty.
        
        Args:
            feat_match: Matching features of shape (B, C_match, H, W) or (B, H*W, C_match)
            feat_geom: Geometric features of shape (B, C_geom, H, W) or (B, H*W, C_geom)
            return_uncertainty: Whether to return uncertainty maps
            
        Returns:
            Dictionary containing:
                - 'flow': Flow map of shape (B, H, W, 2) - (delta_x, delta_y)
                - 'uncertainty': Uncertainty map of shape (B, H, W, 2) - (sigma_x, sigma_y)
                - 'flow_with_uncertainty': Combined tensor of shape (B, H, W, 4)
        """
        # Handle both 2D and sequential inputs
        if feat_match.dim() == 4:  # (B, C, H, W)
            B, C_match, H, W = feat_match.shape
            feat_match = rearrange(feat_match, 'b c h w -> b (h w) c')
            feat_geom = rearrange(feat_geom, 'b c h w -> b (h w) c')
            need_reshape = True
        else:  # (B, L, C)
            B, L, C_match = feat_match.shape
            H = W = int(L ** 0.5)  # Assume square feature maps
            need_reshape = False
        
        # Concatenate matching and geometric features
        combined_feat = torch.cat([feat_match, feat_geom], dim=-1)
        
        # Project to hidden dimension
        h = self.input_proj(combined_feat)
        h = F.gelu(h)
        
        # Process through main network
        h = self.flow_net(h)
        h = self.layer_norm(h)
        
        # Predict flow and uncertainty
        if self.use_separate_heads:
            flow = self.flow_xy_head(h)  # (B, L, 2)
            log_uncertainty = self.uncertainty_head(h)  # (B, L, 2)
            flow_with_uncertainty = torch.cat([flow, log_uncertainty], dim=-1)
        else:
            flow_with_uncertainty = self.flow_head(h)  # (B, L, 4)
            flow = flow_with_uncertainty[..., :2]
            log_uncertainty = flow_with_uncertainty[..., 2:]
        
        # Convert log uncertainty to actual uncertainty (always positive)
        uncertainty = torch.exp(log_uncertainty)
        
        # Reshape back to spatial dimensions if needed
        if need_reshape:
            flow = rearrange(flow, 'b (h w) c -> b h w c', h=H, w=W)
            uncertainty = rearrange(uncertainty, 'b (h w) c -> b h w c', h=H, w=W)
            flow_with_uncertainty = rearrange(flow_with_uncertainty, 'b (h w) c -> b h w c', h=H, w=W)
        
        output = {
            'flow': flow,  # (B, H, W, 2) or (B, L, 2)
            'flow_with_uncertainty': flow_with_uncertainty  # (B, H, W, 4) or (B, L, 4)
        }
        
        if return_uncertainty:
            output['uncertainty'] = uncertainty  # (B, H, W, 2) or (B, L, 2)
            
        return output


class AdaptiveSpanComputer(nn.Module):
    """
    Compute adaptive attention spans from flow and uncertainty predictions.
    
    This module converts flow predictions and uncertainties into attention spans
    for the local Mamba processing in AS-Mamba blocks.
    """
    
    def __init__(
        self,
        base_span: int = 7,
        max_span: int = 15,
        temperature: float = 1.0
    ):
        super().__init__()
        self.base_span = base_span
        self.max_span = max_span
        self.temperature = temperature
    
    def forward(
        self,
        flow: torch.Tensor,
        uncertainty: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive spans from flow and uncertainty.
        
        Args:
            flow: Flow predictions of shape (B, H, W, 2)
            uncertainty: Uncertainty values of shape (B, H, W, 2)
            threshold: Confidence threshold for span adaptation
            
        Returns:
            spans_x: Adaptive spans in x direction (B, H, W)
            spans_y: Adaptive spans in y direction (B, H, W)
        """
        # Normalize uncertainty to [0, 1] range (confidence)
        confidence = 1.0 / (1.0 + uncertainty)  # Higher uncertainty -> lower confidence
        
        # Compute adaptive spans based on confidence
        # High confidence -> smaller span (more certain about correspondence)
        # Low confidence -> larger span (need to search more)
        span_ratio = 1.0 - confidence  # (B, H, W, 2)
        
        # Apply temperature scaling
        span_ratio = torch.sigmoid((span_ratio - threshold) / self.temperature)
        
        # Compute actual spans
        span_range = self.max_span - self.base_span
        spans = self.base_span + span_ratio * span_range  # (B, H, W, 2)
        
        # Round to nearest odd number (for symmetric windows)
        spans = torch.round(spans / 2) * 2 + 1
        spans = torch.clamp(spans, min=self.base_span, max=self.max_span)
        
        spans_x = spans[..., 0].long()  # (B, H, W)
        spans_y = spans[..., 1].long()  # (B, H, W)
        
        return spans_x, spans_y


class HierarchicalFlowPredictor(nn.Module):
    """
    Hierarchical flow predictor that operates at multiple scales.
    
    This is an enhanced version that processes features at different resolutions
    for more robust flow prediction, similar to ASpanFormer's multi-scale approach.
    """
    
    def __init__(
        self,
        d_model_coarse: int = 256,
        d_model_fine: int = 128,
        d_geom: int = 64,
        scales: Tuple[int, ...] = (8, 4, 2)
    ):
        super().__init__()
        self.scales = scales
        
        # Flow predictors for each scale
        self.flow_predictors = nn.ModuleDict({
            f'scale_{s}': FlowPredictor(
                d_model=d_model_coarse if s >= 8 else d_model_fine,
                d_geom=d_geom,
                hidden_dim=128,
                num_layers=2
            )
            for s in scales
        })
        
        # Fusion module to combine multi-scale predictions
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 4)
        )
    
    def forward(
        self,
        feat_match_dict: Dict[int, torch.Tensor],
        feat_geom_dict: Dict[int, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-scale processing.
        
        Args:
            feat_match_dict: Dictionary of matching features at different scales
            feat_geom_dict: Dictionary of geometric features at different scales
            
        Returns:
            Fused flow predictions and uncertainties
        """
        flow_predictions = []
        
        for scale in self.scales:
            if scale in feat_match_dict:
                predictor = self.flow_predictors[f'scale_{scale}']
                pred = predictor(feat_match_dict[scale], feat_geom_dict[scale])
                
                # Upsample to common resolution if needed
                if scale != self.scales[0]:
                    scale_factor = self.scales[0] // scale
                    pred_up = F.interpolate(
                        rearrange(pred['flow_with_uncertainty'], 'b h w c -> b c h w'),
                        scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_up = rearrange(pred_up, 'b c h w -> b h w c')
                    flow_predictions.append(pred_up)
                else:
                    flow_predictions.append(pred['flow_with_uncertainty'])
        
        # Fuse multi-scale predictions
        if len(flow_predictions) > 1:
            combined = torch.cat(flow_predictions, dim=-1)
            fused = self.fusion(combined)
            
            return {
                'flow': fused[..., :2],
                'uncertainty': torch.exp(fused[..., 2:]),
                'flow_with_uncertainty': fused
            }
        else:
            # Single scale
            return {
                'flow': flow_predictions[0][..., :2],
                'uncertainty': torch.exp(flow_predictions[0][..., 2:]),
                'flow_with_uncertainty': flow_predictions[0]
            }