"""
Loss functions for AS-Mamba model.

Extends JamMa's loss with:
1. Flow prediction loss
2. Geometric consistency loss
3. Adaptive span regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from einops import rearrange
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric


class AS_MambaLoss(nn.Module):
    """
    Complete loss function for AS-Mamba model.
    
    Components:
    - Coarse matching loss (from JamMa)
    - Fine matching loss (from JamMa)
    - Flow prediction loss (new)
    - Geometric consistency loss (new)
    - Sub-pixel refinement loss (from JamMa)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_config = config['as_mamba']['loss']
        
        # Loss weights
        self.coarse_weight = self.loss_config['coarse_weight']
        self.fine_weight = self.loss_config['fine_weight']
        self.flow_weight = self.loss_config.get('flow_weight', 0.5)
        self.geom_weight = self.loss_config.get('geom_weight', 0.1)
        self.sub_weight = self.loss_config.get('sub_weight', 1e4)
        
        # Focal loss parameters
        self.focal_alpha = self.loss_config['focal_alpha']
        self.focal_gamma = self.loss_config['focal_gamma']
        self.pos_w = self.loss_config['pos_weight']
        self.neg_w = self.loss_config['neg_weight']
        
    def compute_coarse_loss(self, data, weight=None):
        """
        Coarse-level matching loss (inherited from JamMa).
        
        Args:
            data: Dictionary with conf_matrix_0_to_1, conf_matrix_1_to_0, conf_matrix_gt
            weight: Optional element-wise weights
        """
        conf_matrix_0_to_1 = data["conf_matrix_0_to_1"]
        conf_matrix_1_to_0 = data["conf_matrix_1_to_0"]
        conf_gt = data["conf_matrix_gt"]

        pos_mask = conf_gt == 1
        c_pos_w = self.pos_w
        
        # Handle case with no GT matches
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.

        # Clamp confidences
        conf_matrix_0_to_1 = torch.clamp(conf_matrix_0_to_1, 1e-6, 1-1e-6)
        conf_matrix_1_to_0 = torch.clamp(conf_matrix_1_to_0, 1e-6, 1-1e-6)
        
        # Focal loss
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        
        loss_pos = - alpha * torch.pow(1 - conf_matrix_0_to_1[pos_mask], gamma) * (conf_matrix_0_to_1[pos_mask]).log()
        loss_pos += - alpha * torch.pow(1 - conf_matrix_1_to_0[pos_mask], gamma) * (conf_matrix_1_to_0[pos_mask]).log()
        
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        
        loss_c = c_pos_w * loss_pos.mean()
        return loss_c

    def compute_fine_matching_loss(self, data):
        """
        Fine-level matching loss (inherited from JamMa).
        """
        conf_matrix_fine = data['conf_matrix_fine']
        conf_matrix_f_gt = data['conf_matrix_f_gt']
        
        pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0
        pos_w, neg_w = self.pos_w, self.neg_w

        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_w = 0.

        conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1-1e-6)
        
        # Focal loss
        alpha = self.focal_alpha
        gamma = self.focal_gamma

        loss_pos = - alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma) * (conf_matrix_fine[pos_mask]).log()
        loss_neg = - alpha * torch.pow(conf_matrix_fine[neg_mask], gamma) * (1 - conf_matrix_fine[neg_mask]).log()

        return pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
    
    def compute_flow_loss(self, data):
        """
        Flow prediction loss.
        
        Computes loss between predicted flow and ground truth optical flow
        derived from depth and camera poses.
        """
        if 'flow_map' not in data:
            return torch.tensor(0.0, device=data['feat_8_0'].device)
        
        # Get predicted flow
        flow_pred = data['flow_map']  # (2B, H, W, 4) - includes both images
        B = data['bs']
        
        # Split flow for two images
        if flow_pred.shape[0] == 2 * B:
            flow_pred_0 = flow_pred[:B]  # Flow for image 0
            flow_pred_1 = flow_pred[B:]   # Flow for image 1
        else:
            flow_pred_0 = flow_pred_1 = flow_pred
        
        # Compute ground truth flow from depth and poses
        flow_gt_0, valid_mask_0 = self.compute_gt_flow(
            data, image_idx=0
        )
        flow_gt_1, valid_mask_1 = self.compute_gt_flow(
            data, image_idx=1
        )
        
        # Extract predicted flow components
        flow_xy_0 = flow_pred_0[..., :2]  # (B, H, W, 2)
        flow_uncertainty_0 = torch.exp(flow_pred_0[..., 2:])  # (B, H, W, 2)
        
        flow_xy_1 = flow_pred_1[..., :2]
        flow_uncertainty_1 = torch.exp(flow_pred_1[..., 2:])
        
        # Compute flow loss with uncertainty weighting
        if valid_mask_0.any() and flow_gt_0 is not None:
            # L1 loss weighted by inverse uncertainty
            flow_diff_0 = torch.abs(flow_xy_0 - flow_gt_0)
            weighted_loss_0 = flow_diff_0 / (flow_uncertainty_0 + 1e-8)
            
            # Add uncertainty regularization (prevent trivial solution of infinite uncertainty)
            uncertainty_reg_0 = torch.log(flow_uncertainty_0 + 1e-8)
            
            loss_0 = (weighted_loss_0[valid_mask_0].mean() + 
                     0.1 * uncertainty_reg_0[valid_mask_0].mean())
        else:
            loss_0 = torch.tensor(0.0, device=flow_pred_0.device)
        
        # Same for image 1
        if valid_mask_1.any() and flow_gt_1 is not None:
            flow_diff_1 = torch.abs(flow_xy_1 - flow_gt_1)
            weighted_loss_1 = flow_diff_1 / (flow_uncertainty_1 + 1e-8)
            uncertainty_reg_1 = torch.log(flow_uncertainty_1 + 1e-8)
            
            loss_1 = (weighted_loss_1[valid_mask_1].mean() + 
                     0.1 * uncertainty_reg_1[valid_mask_1].mean())
        else:
            loss_1 = torch.tensor(0.0, device=flow_pred_1.device)
        
        return (loss_0 + loss_1) / 2
    
    def compute_gt_flow(self, data, image_idx=0):
        """
        Compute ground truth optical flow from depth and camera poses.
        
        Args:
            data: Data dictionary
            image_idx: 0 or 1 for which image to compute flow
            
        Returns:
            flow_gt: Ground truth flow (B, H, W, 2)
            valid_mask: Valid flow locations (B, H, W)
        """
        if 'depth0' not in data or 'depth1' not in data:
            return None, torch.zeros(data['bs'], data['h_8'], data['w_8'], 
                                    device=data['feat_8_0'].device, dtype=torch.bool)
        
        B = data['bs']
        H, W = data['h_8'], data['w_8']
        device = data['feat_8_0'].device
        
        # Create grid of pixel coordinates
        grid = create_meshgrid(H, W, device)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        
        # Scale grid to match original image coordinates
        scale = 8  # From H/8 resolution
        grid_scaled = grid * scale
        
        if image_idx == 0:
            depth = F.interpolate(data['depth0'].unsqueeze(1), 
                                size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
            T_0to1 = data['T_0to1']
            K0, K1 = data['K0'], data['K1']
        else:
            depth = F.interpolate(data['depth1'].unsqueeze(1), 
                                size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
            T_0to1 = data['T_1to0']  # Inverse transformation
            K0, K1 = data['K1'], data['K0']
        
        # Valid depth mask
        valid_depth = depth > 0
        
        # Unproject to 3D
        grid_homo = convert_points_to_homogeneous(grid_scaled.reshape(B, -1, 2))  # (B, H*W, 3)
        depth_flat = depth.reshape(B, -1, 1)  # (B, H*W, 1)
        
        # Multiply by depth to get 3D points in camera coordinates
        pts_3d_cam = grid_homo * depth_flat  # (B, H*W, 3)
        
        # Transform to camera coordinates
        pts_3d_cam = torch.bmm(torch.inverse(K0).unsqueeze(0).repeat(B, 1, 1),
                               pts_3d_cam.transpose(1, 2)).transpose(1, 2)
        
        # Transform to other camera
        R = T_0to1[:, :3, :3]  # (B, 3, 3)
        t = T_0to1[:, :3, 3:4]  # (B, 3, 1)
        pts_3d_transformed = torch.bmm(R, pts_3d_cam.transpose(1, 2)) + t  # (B, 3, H*W)
        pts_3d_transformed = pts_3d_transformed.transpose(1, 2)  # (B, H*W, 3)
        
        # Project to image plane
        pts_2d_projected = torch.bmm(K1.unsqueeze(0).repeat(B, 1, 1),
                                     pts_3d_transformed.transpose(1, 2)).transpose(1, 2)
        
        # Normalize by depth
        z = pts_2d_projected[..., 2:3].clamp(min=1e-8)
        pts_2d_normalized = pts_2d_projected[..., :2] / z  # (B, H*W, 2)
        
        # Reshape back to spatial dimensions
        pts_2d_normalized = pts_2d_normalized.reshape(B, H, W, 2)
        
        # Compute flow (in downscaled coordinates)
        flow_gt = (pts_2d_normalized - grid_scaled) / scale  # Scale back to feature resolution
        
        # Compute valid mask
        valid_proj = (pts_2d_normalized[..., 0] >= 0) & \
                    (pts_2d_normalized[..., 0] < W * scale) & \
                    (pts_2d_normalized[..., 1] >= 0) & \
                    (pts_2d_normalized[..., 1] < H * scale)
        
        valid_mask = valid_depth & valid_proj
        
        return flow_gt, valid_mask
    
    def compute_geometric_consistency_loss(self, data):
        """
        Geometric consistency loss for the geometry head.
        
        Ensures that geometric features maintain epipolar constraints.
        """
        if 'feat_geom_0' not in data or 'feat_geom_1' not in data:
            return torch.tensor(0.0, device=data['feat_8_0'].device)
        
        feat_geom_0 = data['feat_geom_0']  # (B, d_geom, H*W)
        feat_geom_1 = data['feat_geom_1']
        
        B = data['bs']
        
        # Reshape to spatial dimensions
        H, W = data['h_8'], data['w_8']
        feat_geom_0 = feat_geom_0.view(B, -1, H, W)
        feat_geom_1 = feat_geom_1.view(B, -1, H, W)
        
        # Compute similarity matrix using geometric features
        feat_geom_0_norm = F.normalize(feat_geom_0, p=2, dim=1)
        feat_geom_1_norm = F.normalize(feat_geom_1, p=2, dim=1)
        
        # Flatten for similarity computation
        feat_geom_0_flat = rearrange(feat_geom_0_norm, 'b c h w -> b (h w) c')
        feat_geom_1_flat = rearrange(feat_geom_1_norm, 'b c h w -> b (h w) c')
        
        # Compute similarity matrix
        sim_matrix = torch.bmm(feat_geom_0_flat, feat_geom_1_flat.transpose(1, 2))  # (B, HW, HW)
        
        # Apply temperature scaling
        temperature = 0.1
        sim_matrix = sim_matrix / temperature
        
        # Get ground truth correspondences from conf_matrix_gt if available
        if 'conf_matrix_gt' in data:
            gt_matrix = data['conf_matrix_gt']  # (B, HW, HW)
            
            # Compute cross-entropy loss
            # Positive pairs (matches)
            pos_mask = gt_matrix > 0
            if pos_mask.any():
                pos_sim = sim_matrix[pos_mask]
                # Negative log-likelihood for positive pairs
                pos_loss = -F.logsigmoid(pos_sim).mean()
            else:
                pos_loss = torch.tensor(0.0, device=sim_matrix.device)
            
            # Negative pairs (non-matches)
            neg_mask = gt_matrix == 0
            # Sample negative pairs to avoid imbalance
            num_neg_samples = min(pos_mask.sum() * 5, neg_mask.sum())
            if num_neg_samples > 0 and neg_mask.any():
                neg_sim_all = sim_matrix[neg_mask]
                # Random sampling of negatives
                perm = torch.randperm(neg_sim_all.shape[0])[:num_neg_samples]
                neg_sim = neg_sim_all[perm]
                # Negative log-likelihood for negative pairs
                neg_loss = -F.logsigmoid(-neg_sim).mean()
            else:
                neg_loss = torch.tensor(0.0, device=sim_matrix.device)
            
            geom_consistency_loss = (pos_loss + neg_loss) / 2
        else:
            # If no GT available, use self-consistency
            # Enforce that geometric features are distinctive
            geom_consistency_loss = self.compute_distinctiveness_loss(sim_matrix)
        
        return geom_consistency_loss
    
    def compute_distinctiveness_loss(self, sim_matrix):
        """
        Encourage geometric features to be distinctive.
        Each point should have high similarity with few other points.
        """
        B, N, M = sim_matrix.shape
        
        # Compute entropy of similarity distributions
        sim_probs_0to1 = F.softmax(sim_matrix, dim=2)
        sim_probs_1to0 = F.softmax(sim_matrix, dim=1)
        
        # Entropy (we want low entropy = peaky distribution)
        entropy_0to1 = -(sim_probs_0to1 * torch.log(sim_probs_0to1 + 1e-8)).sum(dim=2).mean()
        entropy_1to0 = -(sim_probs_1to0 * torch.log(sim_probs_1to0 + 1e-8)).sum(dim=1).mean()
        
        # Negative entropy as loss (minimize entropy)
        return (entropy_0to1 + entropy_1to0) / 2
    
    def compute_sub_pixel_loss(self, data):
        """
        Sub-pixel refinement loss (inherited from JamMa).
        """
        if 'mkpts0_f_train' not in data or 'mkpts1_f_train' not in data:
            return torch.tensor(0.0, device=data['feat_8_0'].device)
        
        # Compute essential matrix
        Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
        E_mat = Tx @ data['T_0to1'][:, :3, :3]
        
        m_bids = data['m_bids']
        pts0 = data['mkpts0_f_train']
        pts1 = data['mkpts1_f_train']
        
        # Symmetric epipolar distance
        sym_dist = self._symmetric_epipolar_distance(
            pts0, pts1, E_mat[m_bids], data['K0'][m_bids], data['K1'][m_bids]
        )
        
        # Filter high-error matches
        if len(sym_dist) == 0:
            return torch.tensor(0.0, device=data['feat_8_0'].device)
        
        loss = sym_dist[sym_dist < 1e-4]
        if len(loss) == 0:
            loss = sym_dist * 1e-9
        
        return loss.mean()
    
    def _symmetric_epipolar_distance(self, pts0, pts1, E, K0, K1):
        """Compute symmetric epipolar distance."""
        # Normalize points
        pts0 = (pts0 - K0[:, [0, 1], [2, 2]]) / K0[:, [0, 1], [0, 1]]
        pts1 = (pts1 - K1[:, [0, 1], [2, 2]]) / K1[:, [0, 1], [0, 1]]
        pts0 = convert_points_to_homogeneous(pts0)
        pts1 = convert_points_to_homogeneous(pts1)

        # Compute epipolar lines
        Ep0 = (pts0[:,None,:] @ E.transpose(-2,-1)).squeeze(1)
        p1Ep0 = torch.sum(pts1 * Ep0, -1)
        Etp1 = (pts1[:,None,:] @ E).squeeze(1)

        # Symmetric distance
        d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2 + 1e-9) + 
                       1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2 + 1e-9))
        return d
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """Compute element-wise weights for coarse loss."""
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * 
                       data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight
    
    def forward(self, data):
        """
        Compute total loss for AS-Mamba.
        
        Args:
            data: Dictionary containing model outputs and ground truth
            
        Updates:
            data['loss']: Total loss
            data['loss_scalars']: Individual loss components for logging
        """
        loss_scalars = {}
        
        # 1. Coarse-level matching loss
        c_weight = self.compute_c_weight(data)
        loss_c = self.compute_coarse_loss(data, weight=c_weight)
        loss_c *= self.coarse_weight
        loss = loss_c
        loss_scalars['loss_c'] = loss_c.clone().detach().cpu()
        
        # 2. Fine-level matching loss
        if 'conf_matrix_fine' in data:
            loss_f = self.compute_fine_matching_loss(data)
            loss_f *= self.fine_weight
            loss = loss + loss_f
            loss_scalars['loss_f'] = loss_f.clone().detach().cpu()
        else:
            loss_scalars['loss_f'] = torch.tensor(0.0)
        
        # 3. Flow prediction loss (NEW)
        loss_flow = self.compute_flow_loss(data)
        if loss_flow is not None and loss_flow > 0:
            loss_flow *= self.flow_weight
            loss = loss + loss_flow
            loss_scalars['loss_flow'] = loss_flow.clone().detach().cpu()
        else:
            loss_scalars['loss_flow'] = torch.tensor(0.0)
        
        # 4. Geometric consistency loss (NEW)
        loss_geom = self.compute_geometric_consistency_loss(data)
        if loss_geom is not None and loss_geom > 0:
            loss_geom *= self.geom_weight
            loss = loss + loss_geom
            loss_scalars['loss_geom'] = loss_geom.clone().detach().cpu()
        else:
            loss_scalars['loss_geom'] = torch.tensor(0.0)
        
        # 5. Sub-pixel refinement loss
        loss_sub = self.compute_sub_pixel_loss(data)
        if loss_sub is not None and loss_sub > 0:
            loss_sub *= self.sub_weight
            loss = loss + loss_sub
            loss_scalars['loss_sub'] = loss_sub.clone().detach().cpu()
        else:
            loss_scalars['loss_sub'] = torch.tensor(0.0)
        
        # Update data with losses
        loss_scalars['loss'] = loss.clone().detach().cpu()
        data.update({
            'loss': loss,
            'loss_scalars': loss_scalars
        })
        
        return loss


def create_meshgrid(H, W, device='cpu'):
    """Create a meshgrid of pixel coordinates."""
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    grid = torch.stack([x, y], dim=-1)  # (H, W, 2)
    return grid