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


"""
AS-Mamba Loss Function

Combines the strengths of ASpanFormer and JamMa:
1. Flow prediction loss with uncertainty (from ASpanFormer)
2. Symmetric epipolar distance (from JamMa) 
3. Hierarchical matching losses for global and local paths
4. Geometric consistency constraints

Design Rationale:
- Flow loss is ESSENTIAL for adaptive span learning
- Epipolar constraint ensures geometric consistency
- Multi-scale supervision leverages hierarchical architecture
- Uncertainty weighting improves robustness
"""

class ASMambaLoss(nn.Module):
    """
    Comprehensive loss function for AS-Mamba architecture.
    
    Loss components:
    1. Flow prediction loss (adaptive span guidance)
    2. Coarse matching loss (global correspondence)
    3. Fine matching loss (local refinement)
    4. Epipolar geometric loss (geometric consistency)
    5. Multi-scale consistency loss (hierarchical supervision)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_config = config['as_mamba']['loss']
        
        # Matching configuration
        self.match_type = self.config['as_mamba']['match_coarse'].get('match_type', 'dual_softmax')
        self.sparse_spvs = self.config['as_mamba']['match_coarse'].get('sparse_spvs', False)
        
        # Loss weights
        self.flow_weight = self.loss_config['flow_weight']
        self.coarse_weight = self.loss_config['coarse_weight']
        self.fine_weight = self.loss_config['fine_weight']
        self.epipolar_weight = self.loss_config.get('epipolar_weight', 0.1)
        self.multiscale_weight = self.loss_config.get('multiscale_weight', 0.05)
        
        # Coarse-level weights
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        
        # Fine-level configuration
        self.fine_type = self.loss_config.get('fine_type', 'l2')
        self.correct_thr = self.loss_config.get('fine_correct_thr', 0.5)
        
        # Focal loss parameters
        self.focal_alpha = self.loss_config.get('focal_alpha', 0.25)
        self.focal_gamma = self.loss_config.get('focal_gamma', 2.0)
        
    def compute_flow_loss(self, flow_predictions, coarse_corr_gt, hw0, hw1):
        """
        Flow prediction loss with uncertainty weighting.
        
        Critical for AS-Mamba's adaptive span mechanism.
        Adapted from ASpanFormer but enhanced for hierarchical flow.
        
        Args:
            flow_predictions: List of [flow_0to1, flow_1to0] for each AS-Mamba block
                Each flow: (num_blocks, B, H, W, 4) where last dim is [dx, dy, ux, uy]
            coarse_corr_gt: [batch_ids, left_ids, right_ids] ground truth correspondences
            hw0, hw1: (H, W) dimensions for image 0 and 1
            
        Returns:
            Flow loss tensor
        """
        batch_ids, self_ids_0, cross_ids_1 = coarse_corr_gt
        h0, w0 = hw0
        h1, w1 = hw1
        
        total_loss = 0.0
        loss_breakdown = []
        
        for block_idx, (flow_0to1, flow_1to0) in enumerate(flow_predictions):
            # Forward flow loss (image 0 -> 1)
            loss_forward = self._flow_loss_worker(
                flow_0to1, batch_ids, self_ids_0, cross_ids_1, w1
            )
            
            # Backward flow loss (image 1 -> 0)  
            loss_backward = self._flow_loss_worker(
                flow_1to0, batch_ids, cross_ids_1, self_ids_0, w0
            )
            
            # Average bidirectional flow loss
            block_loss = (loss_forward + loss_backward) / 2.0
            total_loss += block_loss
            loss_breakdown.append(block_loss)
            
        return total_loss, loss_breakdown
    
    def _flow_loss_worker(self, flow, batch_ids, self_ids, cross_ids, target_w):
        """
        Compute flow loss for one direction with uncertainty weighting.
        
        Loss formulation: L = u + exp(-u) * ||flow_pred - flow_gt||^2
        where u is the predicted uncertainty (log variance).
        
        This encourages:
        - Low uncertainty + accurate flow -> low loss
        - High uncertainty for difficult regions -> moderate loss
        - Prevents trivial solution (infinite uncertainty) via u term
        """
        num_blocks, bs = flow.shape[0], flow.shape[1]
        flow = flow.view(num_blocks, bs, -1, 4)
        
        # Compute ground truth flow
        gt_flow = torch.stack([
            cross_ids % target_w,
            cross_ids // target_w
        ], dim=1).float()  # (N, 2)
        
        block_losses = []
        for block_idx in range(num_blocks):
            # Extract predicted flow and uncertainty at supervision points
            pred_flow = flow[block_idx][batch_ids, self_ids, :2]  # (N, 2)
            pred_uncertainty = flow[block_idx][batch_ids, self_ids, 2:]  # (N, 2)
            
            # L2 distance between predicted and GT flow
            flow_error = (gt_flow - pred_flow) ** 2  # (N, 2)
            
            # Uncertainty-weighted loss (separate for x and y)
            loss = pred_uncertainty + torch.exp(-pred_uncertainty) * flow_error
            block_losses.append(loss.mean())
        
        return torch.stack(block_losses).mean() * self.flow_weight
    
    def compute_coarse_loss(self, conf_matrix, conf_matrix_gt, weight=None):
        """
        Coarse-level matching loss using Focal Loss.
        
        Enhanced from both ASpanFormer and JamMa:
        - Focal loss for handling class imbalance
        - Supports both dense and sparse supervision
        - Bidirectional consistency awareness
        
        Args:
            conf_matrix: Confidence matrix (N, HW0, HW1) or with dustbin
            conf_matrix_gt: Ground truth (N, HW0, HW1)
            weight: Optional mask for valid regions
        """
        pos_mask = conf_matrix_gt == 1
        neg_mask = conf_matrix_gt == 0
        
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        
        # Handle corner cases
        if not pos_mask.any():
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
            
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.
        
        # Clamp confidence for numerical stability
        conf_matrix = torch.clamp(conf_matrix, 1e-6, 1 - 1e-6)
        
        # Focal loss computation
        if self.sparse_spvs and self.match_type == 'sinkhorn':
            # Sparse supervision with dustbin
            pos_conf = conf_matrix[:, :-1, :-1][pos_mask]
            loss_pos = -self.focal_alpha * \
                       torch.pow(1 - pos_conf, self.focal_gamma) * pos_conf.log()
            
            # Negative samples from dustbin rows/columns
            neg0 = conf_matrix_gt.sum(-1) == 0
            neg1 = conf_matrix_gt.sum(1) == 0
            neg_conf = torch.cat([
                conf_matrix[:, :-1, -1][neg0],
                conf_matrix[:, -1, :-1][neg1]
            ], dim=0)
            loss_neg = -self.focal_alpha * \
                       torch.pow(1 - neg_conf, self.focal_gamma) * neg_conf.log()
            
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                neg_w0 = (weight.sum(-1) != 0)[neg0]
                neg_w1 = (weight.sum(1) != 0)[neg1]
                neg_mask_w = torch.cat([neg_w0, neg_w1], dim=0)
                loss_neg = loss_neg[neg_mask_w]
            
            loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        else:
            # Dense supervision
            loss_pos = -self.focal_alpha * \
                       torch.pow(1 - conf_matrix[pos_mask], self.focal_gamma) * \
                       conf_matrix[pos_mask].log()
            loss_neg = -self.focal_alpha * \
                       torch.pow(conf_matrix[neg_mask], self.focal_gamma) * \
                       (1 - conf_matrix[neg_mask]).log()
            
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            
            loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        
        return loss
    
    def compute_fine_loss(self, expec_f, expec_f_gt, expec_f_gt_mask):
        """
        Fine-level matching loss with uncertainty.
        
        Supports two modes:
        1. L2 with std: Uncertainty-weighted L2 distance
        2. L2: Simple L2 distance
        
        Args:
            expec_f: Predicted fine-level offsets (M, 2) or (M, 3) with std
            expec_f_gt: Ground truth offsets (M, 2)
        """
        if expec_f_gt_mask.sum() == 0:
            return torch.tensor(0.0, device=expec_f.device)
        
        expec_f_filtered_by_mask = expec_f[expec_f_gt_mask]
        expec_f_gt_filtered_by_mask = expec_f_gt[expec_f_gt_mask]
        
        nan_mask = ~torch.isnan(expec_f_gt_filtered_by_mask).any(dim=-1)
        if nan_mask.sum() == 0:
            return torch.tensor(0.0, device=expec_f.device)
        
        expec_f_final = expec_f_filtered_by_mask[nan_mask]
        expec_f_gt_final = expec_f_gt_filtered_by_mask[nan_mask]
        
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError(f"Fine loss type {self.fine_type} not implemented")
    
    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """Simple L2 fine-level loss."""
        # logger.debug("Using simple L2 fine-level loss.")
        correct_mask = torch.linalg.norm(
            expec_f_gt, ord=float('inf'), dim=1
        ) < self.correct_thr

        # Extract std and compute weights
        std = expec_f[:, 2]
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()

        # if not correct_mask.any():
        #     logger.warning("No correct fine matches - returning zero loss for fine-level (graph preserved).")
            
        #     # expec_f_gt = torch.zeros(1, 2, device=expec_f.device)
        #     # expec_f = torch.zeros(1, 3, device=expec_f.device)
        #     # correct_mask = torch.ones(1, dtype=torch.bool, device=expec_f.device)
        #     return expec_f.sum() * 0.0
        
        if correct_mask.sum() == 0:
            if self.training:
                logger.warning("No correct fine matches - returning zero loss for fine-level (graph preserved).")
                return (expec_f.sum() * 0.0) 
                
            # if self.training:
            #     logger.warning("No correct fine matches - assigning dummy supervision")
            #     correct_mask[0] = True
            #     weight[0] = 0.
            else:
                return None
        
        # flow_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        # return flow_l2.mean()
        flow_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask][:, :2]) ** 2).sum(-1)
        return flow_l2.mean()
    
    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Fine-level loss with uncertainty weighting.
        
        Similar to flow loss: uses inverse std as weight.
        Prevents trivial solutions by detaching weight from gradient.
        """
        correct_mask = torch.linalg.norm(
            expec_f_gt, ord=float('inf'), dim=1
        ) < self.correct_thr
        
        # Extract std and compute weights
        std = expec_f[:, 2]
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()
        
        if not correct_mask.any():
            if self.training:
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None
        
        # Weighted L2 loss
        flow_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (flow_l2 * weight[correct_mask]).mean()
        
        return loss
    
    def compute_geometric_loss(self, geom_outputs_0, geom_outputs_1, conf_gt, weight):
        """
        Compute geometric consistency loss for all geometry heads.
        Uses coarse matching loss logic on geometric features.
        """
        total_geom_loss = 0.0
        
        if not geom_outputs_0:
            return torch.tensor(0.0, device=conf_gt.device)
            
        for feat_g_0, feat_g_1 in zip(geom_outputs_0, geom_outputs_1):
            # feat_g (B, C_geom, HW) -> (B, HW, C_geom)
            feat_g_0_flat = feat_g_0.transpose(1, 2)
            feat_g_1_flat = feat_g_1.transpose(1, 2)
            
            feat_g_0_flat = F.normalize(feat_g_0_flat, p=2, dim=-1)
            feat_g_1_flat = F.normalize(feat_g_1_flat, p=2, dim=-1)

            sim_matrix = torch.einsum('bic,bjc->bij', feat_g_0_flat, feat_g_1_flat)
            

            sim_matrix = sim_matrix / 0.1 
            
            total_geom_loss += self.compute_coarse_loss(sim_matrix, conf_gt, weight)
            
        return total_geom_loss / len(geom_outputs_0)
    
    def compute_epipolar_loss(self, data):
        """
        Symmetric epipolar distance loss (from JamMa).
        
        WHY THIS IS IMPORTANT FOR AS-MAMBA:
        - Ensures geometric consistency of matches
        - Complements flow-based supervision with hard geometric constraints
        - Particularly useful when ground truth flow may be noisy
        - Acts as regularization for the learned geometric features
        
        Args:
            data: Dictionary containing matched points and camera parameters
        """
        if 'mkpts0_f' not in data or 'T_0to1' not in data:
            return None
        
        valid_mask = data['mconf_f'] > 0.0
        # logger.debug(f"valid_mask sum: {valid_mask.sum().item()}")
        # logger.debug(f"valid_mask shape: {valid_mask.shape}")
        num_valid_matches = valid_mask.sum()
        # logger.debug(f"Number of valid matches for epipolar loss: {num_valid_matches.item()}")

        if num_valid_matches == 0:
            logger.debug("No valid matches found, skipping epipolar loss.")
            return torch.tensor(0.0, device=data['mconf_f'].device)
        
        # logger.debug(f"Data keys: {list(data.keys())}")
        
        m_bids = data['b_ids_fine']
        # logger.debug(f"Computing epipolar loss for {len(m_bids)} matches.")

        pts0 = data['mkpts0_f'][valid_mask]
        pts1 = data['mkpts1_f'][valid_mask]

        # logger.debug(f"pts0 shape: {pts0.shape}")
        # logger.debug(f"pts1 shape: {pts1.shape}")

        K0 = data['K0'][m_bids][valid_mask]
        K1 = data['K1'][m_bids][valid_mask]
        # logger.debug(f"K0 shape after masking: {K0.shape}") 
        # logger.debug(f"K1 shape after masking: {K1.shape}")

        T_0to1 = data['T_0to1'][m_bids][valid_mask]

        # logger.debug(f"T_0to1 shape after masking: {T_0to1.shape}")
        
        # Compute essential matrix
        # Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
        # E_mat = Tx @ data['T_0to1'][:, :3, :3]
        T_0to1_filtered = data['T_0to1'][m_bids][valid_mask]
        # logger.debug(f"T_0to1_filtered shape: {T_0to1_filtered.shape}")

        Tx = numeric.cross_product_matrix(T_0to1_filtered[:, :3, 3])
        # logger.debug(f"Tx shape: {Tx.shape}")
        E_mat = Tx @ T_0to1_filtered[:, :3, :3]
        # logger.debug(f"E_mat shape: {E_mat.shape}")

        # Compute symmetric epipolar distance
        # sym_dist = self._symmetric_epipolar_distance(
        #     pts0, pts1, E_mat[m_bids], 
        #     data['K0'][m_bids], data['K1'][m_bids]
        # )
        sym_dist = self._symmetric_epipolar_distance(
            pts0, pts1, E_mat, 
            K0, K1
        )
        
        # Filter outliers (only train on approximately correct matches)
        if len(sym_dist) == 0:
            return None
        
        # Use only inliers for training
        inlier_mask = sym_dist < 1e-4
        if inlier_mask.sum() == 0:
            return sym_dist.mean() * 1e-9  # Very small loss if no inliers
        
        return sym_dist[inlier_mask].mean()
    
    def _symmetric_epipolar_distance(self, pts0, pts1, E, K0, K1):
        """
        Compute symmetric epipolar distance.
        
        Distance = (x1^T E x0)^2 * (1/||Fx0||^2 + 1/||F^T x1||^2)
        
        This is a biased but differentiable approximation of reprojection error.
        """
        # Normalize by intrinsics
        # logger.debug(f"K0 shape: {K0.shape}, K1 shape: {K1.shape}")
        # logger.debug(f"Ko0: {K0}, K1: {K1}")
        # logger.debug(f"E shape: {E.shape}")
        # logger.debug(f"E: {E}")
        # logger.debug(f"pts0 shape: {pts0.shape}, pts1 shape: {pts1.shape}")
        pts0 = (pts0 - K0[:, [0, 1], [2, 2]]) / K0[:, [0, 1], [0, 1]]
        pts1 = (pts1 - K1[:, [0, 1], [2, 2]]) / K1[:, [0, 1], [0, 1]]

        # logger.debug(f"pts0: {pts0}, pts1: {pts1}")
        # logger.debug(f"pts0 shape: {pts0.shape}, pts1 shape: {pts1.shape}")
        
        # Convert to homogeneous
        pts0 = convert_points_to_homogeneous(pts0)
        pts1 = convert_points_to_homogeneous(pts1)
        
        # Compute epipolar lines
        Ep0 = (pts0[:, None, :] @ E.transpose(-2, -1)).squeeze(1)  # (N, 3)
        p1Ep0 = torch.sum(pts1 * Ep0, -1)  # (N,)
        Etp1 = (pts1[:, None, :] @ E).squeeze(1)  # (N, 3)
        
        # Symmetric distance
        d = p1Ep0 ** 2 * (
            1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2 + 1e-9) +
            1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2 + 1e-9)
        )
        
        return d
    
    def compute_multiscale_consistency_loss(self, data):
        """
        Multi-scale consistency loss for hierarchical processing.
        
        WHY THIS MATTERS FOR AS-MAMBA:
        - AS-Mamba has global (downsampled) and local (full-res) paths
        - This loss ensures they produce consistent predictions
        - Improves feature learning at different scales
        - Leverages the hierarchical Mamba architecture
        
        Args:
            data: Dictionary containing global and local feature predictions
        """
        if 'global_matches' not in data or 'local_matches' not in data:
            return None
        
        global_conf = data['global_matches']  # Upsampled to full resolution
        local_conf = data['local_matches']
        
        # KL divergence between global and local predictions
        # Encourages consistency while allowing local refinement
        kl_loss = nn.functional.kl_div(
            torch.log(local_conf + 1e-9),
            global_conf,
            reduction='batchmean'
        )
        
        return kl_loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """Compute element-wise weights for valid regions."""
        if 'mask0' in data:
            c_weight = (
                data['mask0'].flatten(-2)[..., None] * 
                data['mask1'].flatten(-2)[:, None]
            ).float()
        else:
            c_weight = None
        return c_weight
    
    def forward(self, data):
        """
        Compute total loss for AS-Mamba.
        
        Loss composition:
        1. Flow loss (CRITICAL for adaptive spans) - highest weight
        2. Coarse matching loss (global correspondence)
        3. Fine matching loss (local refinement)
        4. Epipolar loss (geometric consistency)
        5. Multi-scale consistency (hierarchical supervision)
        
        Args:
            data: Dictionary containing predictions and ground truth
            
        Returns:
            Updates data with 'loss' and 'loss_scalars'
        """
        loss_scalars = {}
        total_loss = 0.0
        
        # 0. Compute validity weights
        c_weight = self.compute_c_weight(data)
        
        # 1. FLOW LOSS - Most important for AS-Mamba
        if 'predict_flow' in data:
            loss_flow, flow_breakdown = self.compute_flow_loss(
                data['predict_flow'],
                [data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']],
                data['hw0_c'],
                data['hw1_c']
            )
            total_loss += loss_flow
            loss_scalars['loss_flow'] = loss_flow.clone().detach().cpu()
            
            # Log per-block flow losses
            for idx, block_loss in enumerate(flow_breakdown):
                loss_scalars[f'loss_flow_block_{idx}'] = block_loss.clone().detach().cpu()
        
        # 2. COARSE MATCHING LOSS
        conf_key = 'conf_matrix_with_bin' if self.sparse_spvs and self.match_type == 'sinkhorn' \
                   else 'conf_matrix'
        
        if conf_key in data:
            loss_c = self.compute_coarse_loss(
                data[conf_key],
                data['conf_matrix_gt'],
                weight=c_weight
            )
            total_loss += loss_c * self.coarse_weight
            loss_scalars['loss_c'] = loss_c.clone().detach().cpu()
        
        # 3. FINE MATCHING LOSS
        # logger.debug("Computing fine loss...")
        # logger.debug(f"Data keys for fine loss: {list(data.keys())}")
        # if 'expec_f' in data:
        #     logger.debug(f"Computing fine loss using type: {self.fine_type}")
        #     loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if ('expec_f' in data and 'expec_f_gt' in data and 'expec_f_gt_mask' in data and
            data['expec_f'].numel() > 0 and data['expec_f_gt'].numel() > 0):
            loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'], data['expec_f_gt_mask'])
            if loss_f is not None:
                total_loss += loss_f * self.fine_weight
                loss_scalars['loss_f'] = loss_f.clone().detach().cpu()
            else:
                loss_scalars['loss_f'] = torch.tensor(1.0)
        
        # 4. EPIPOLAR GEOMETRIC LOSS
        if self.epipolar_weight > 0:
            loss_epi = self.compute_epipolar_loss(data)
            if loss_epi is not None:
                total_loss += loss_epi * self.epipolar_weight
                loss_scalars['loss_epi'] = loss_epi.clone().detach().cpu()
            else:
                loss_scalars['loss_epi'] = torch.tensor(0.0)
        
        # 2025-10-24 ADD
        # 5. GEOMETRIC CONSISTENCY LOSS
        if 'geom_outputs_0' in data and self.loss_config['geom_weight'] > 0:
            loss_g = self.compute_geometric_loss(
                data['geom_outputs_0'],
                data['geom_outputs_1'],
                data['conf_matrix_gt'],
                c_weight
            )
            if not torch.isnan(loss_g) and not torch.isinf(loss_g):
                total_loss += loss_g * self.loss_config['geom_weight']
                loss_scalars['loss_geom'] = loss_g.clone().detach().cpu()
            else:
                loss_scalars['loss_geom'] = torch.tensor(0.0)
                if self.training:
                     logger.warning("L_geom resulted in NaN/Inf and was skipped.")
        else:
            loss_scalars['loss_geom'] = torch.tensor(0.0)
        
        # 6. MULTI-SCALE CONSISTENCY LOSS
        if self.multiscale_weight > 0:
            loss_ms = self.compute_multiscale_consistency_loss(data)
            if loss_ms is not None:
                total_loss += loss_ms * self.multiscale_weight
                loss_scalars['loss_ms'] = loss_ms.clone().detach().cpu()
            else:
                loss_scalars['loss_ms'] = torch.tensor(0.0)
        
        # Update data
        loss_scalars['loss'] = total_loss.clone().detach().cpu()
        data.update({
            'loss': total_loss,
            'loss_scalars': loss_scalars
        })
        
        return data
    
# class AS_MambaLoss(nn.Module):
#     """
#     Complete loss function for AS-Mamba model.
    
#     Components:
#     - Coarse matching loss (from JamMa)
#     - Fine matching loss (from JamMa)
#     - Flow prediction loss (new)
#     - Geometric consistency loss (new)
#     - Sub-pixel refinement loss (from JamMa)
#     """
    
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.loss_config = config['as_mamba']['loss']
        
#         # Loss weights
#         self.coarse_weight = self.loss_config['coarse_weight']
#         self.fine_weight = self.loss_config['fine_weight']
#         self.flow_weight = self.loss_config.get('flow_weight', 0.5)
#         self.geom_weight = self.loss_config.get('geom_weight', 0.1)
#         self.sub_weight = self.loss_config.get('sub_weight', 1e4)
        
#         # Focal loss parameters
#         self.focal_alpha = self.loss_config['focal_alpha']
#         self.focal_gamma = self.loss_config['focal_gamma']
#         self.pos_w = self.loss_config['pos_weight']
#         self.neg_w = self.loss_config['neg_weight']
        
#     def compute_coarse_loss(self, data, weight=None):
#         """
#         Coarse-level matching loss (inherited from JamMa).
        
#         Args:
#             data: Dictionary with conf_matrix_0_to_1, conf_matrix_1_to_0, conf_matrix_gt
#             weight: Optional element-wise weights
#         """
#         conf_matrix_0_to_1 = data["conf_matrix_0_to_1"]
#         conf_matrix_1_to_0 = data["conf_matrix_1_to_0"]
#         conf_gt = data["conf_matrix_gt"]

#         pos_mask = conf_gt == 1
#         c_pos_w = self.pos_w
        
#         # Handle case with no GT matches
#         if not pos_mask.any():
#             pos_mask[0, 0, 0] = True
#             if weight is not None:
#                 weight[0, 0, 0] = 0.
#             c_pos_w = 0.

#         # Clamp confidences
#         conf_matrix_0_to_1 = torch.clamp(conf_matrix_0_to_1, 1e-6, 1-1e-6)
#         conf_matrix_1_to_0 = torch.clamp(conf_matrix_1_to_0, 1e-6, 1-1e-6)
        
#         # Focal loss
#         alpha = self.focal_alpha
#         gamma = self.focal_gamma
        
#         loss_pos = - alpha * torch.pow(1 - conf_matrix_0_to_1[pos_mask], gamma) * (conf_matrix_0_to_1[pos_mask]).log()
#         loss_pos += - alpha * torch.pow(1 - conf_matrix_1_to_0[pos_mask], gamma) * (conf_matrix_1_to_0[pos_mask]).log()
        
#         if weight is not None:
#             loss_pos = loss_pos * weight[pos_mask]
        
#         loss_c = c_pos_w * loss_pos.mean()
#         return loss_c

#     def compute_fine_matching_loss(self, data):
#         """
#         Fine-level matching loss (inherited from JamMa).
#         """
#         conf_matrix_fine = data['conf_matrix_fine']
#         conf_matrix_f_gt = data['conf_matrix_f_gt']
        
#         pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0
#         pos_w, neg_w = self.pos_w, self.neg_w

#         if not pos_mask.any():
#             pos_mask[0, 0, 0] = True
#             pos_w = 0.
#         if not neg_mask.any():
#             neg_mask[0, 0, 0] = True
#             neg_w = 0.

#         conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1-1e-6)
        
#         # Focal loss
#         alpha = self.focal_alpha
#         gamma = self.focal_gamma

#         loss_pos = - alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma) * (conf_matrix_fine[pos_mask]).log()
#         loss_neg = - alpha * torch.pow(conf_matrix_fine[neg_mask], gamma) * (1 - conf_matrix_fine[neg_mask]).log()

#         return pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
    
#     def compute_flow_loss(self, data):
#         """
#         Flow prediction loss.
        
#         Computes loss between predicted flow and ground truth optical flow
#         derived from depth and camera poses.
#         """
#         if 'flow_map' not in data:
#             return torch.tensor(0.0, device=data['feat_8_0'].device)
        
#         # Get predicted flow
#         flow_pred = data['flow_map']  # (2B, H, W, 4) - includes both images
#         B = data['bs']
        
#         # Split flow for two images
#         if flow_pred.shape[0] == 2 * B:
#             flow_pred_0 = flow_pred[:B]  # Flow for image 0
#             flow_pred_1 = flow_pred[B:]   # Flow for image 1
#         else:
#             flow_pred_0 = flow_pred_1 = flow_pred
        
#         # Compute ground truth flow from depth and poses
#         flow_gt_0, valid_mask_0 = self.compute_gt_flow(
#             data, image_idx=0
#         )
#         flow_gt_1, valid_mask_1 = self.compute_gt_flow(
#             data, image_idx=1
#         )
        
#         # Extract predicted flow components
#         flow_xy_0 = flow_pred_0[..., :2]  # (B, H, W, 2)
#         flow_uncertainty_0 = torch.exp(flow_pred_0[..., 2:])  # (B, H, W, 2)
        
#         flow_xy_1 = flow_pred_1[..., :2]
#         flow_uncertainty_1 = torch.exp(flow_pred_1[..., 2:])
        
#         # Compute flow loss with uncertainty weighting
#         if valid_mask_0.any() and flow_gt_0 is not None:
#             # L1 loss weighted by inverse uncertainty
#             flow_diff_0 = torch.abs(flow_xy_0 - flow_gt_0)
#             weighted_loss_0 = flow_diff_0 / (flow_uncertainty_0 + 1e-8)
            
#             # Add uncertainty regularization (prevent trivial solution of infinite uncertainty)
#             uncertainty_reg_0 = torch.log(flow_uncertainty_0 + 1e-8)
            
#             loss_0 = (weighted_loss_0[valid_mask_0].mean() + 
#                      0.1 * uncertainty_reg_0[valid_mask_0].mean())
#         else:
#             loss_0 = torch.tensor(0.0, device=flow_pred_0.device)
        
#         # Same for image 1
#         if valid_mask_1.any() and flow_gt_1 is not None:
#             flow_diff_1 = torch.abs(flow_xy_1 - flow_gt_1)
#             weighted_loss_1 = flow_diff_1 / (flow_uncertainty_1 + 1e-8)
#             uncertainty_reg_1 = torch.log(flow_uncertainty_1 + 1e-8)
            
#             loss_1 = (weighted_loss_1[valid_mask_1].mean() + 
#                      0.1 * uncertainty_reg_1[valid_mask_1].mean())
#         else:
#             loss_1 = torch.tensor(0.0, device=flow_pred_1.device)
        
#         return (loss_0 + loss_1) / 2
    
#     def compute_gt_flow(self, data, image_idx=0):
#         """
#         Compute ground truth optical flow from depth and camera poses.
        
#         Args:
#             data: Data dictionary
#             image_idx: 0 or 1 for which image to compute flow
            
#         Returns:
#             flow_gt: Ground truth flow (B, H, W, 2)
#             valid_mask: Valid flow locations (B, H, W)
#         """
#         if 'depth0' not in data or 'depth1' not in data:
#             return None, torch.zeros(data['bs'], data['h_8'], data['w_8'], 
#                                     device=data['feat_8_0'].device, dtype=torch.bool)
        
#         B = data['bs']
#         H, W = data['h_8'], data['w_8']
#         device = data['feat_8_0'].device
        
#         # Create grid of pixel coordinates
#         grid = create_meshgrid(H, W, device)  # (H, W, 2)
#         grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        
#         # Scale grid to match original image coordinates
#         scale = 8  # From H/8 resolution
#         grid_scaled = grid * scale
        
#         if image_idx == 0:
#             depth = F.interpolate(data['depth0'].unsqueeze(1), 
#                                 size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
#             T_0to1 = data['T_0to1']
#             K0, K1 = data['K0'], data['K1']
#         else:
#             depth = F.interpolate(data['depth1'].unsqueeze(1), 
#                                 size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
#             T_0to1 = data['T_1to0']  # Inverse transformation
#             K0, K1 = data['K1'], data['K0']
        
#         # Valid depth mask
#         valid_depth = depth > 0
        
#         # Unproject to 3D
#         grid_homo = convert_points_to_homogeneous(grid_scaled.reshape(B, -1, 2))  # (B, H*W, 3)
#         depth_flat = depth.reshape(B, -1, 1)  # (B, H*W, 1)
        
#         # Multiply by depth to get 3D points in camera coordinates
#         pts_3d_cam = grid_homo * depth_flat  # (B, H*W, 3)
        
#         # Transform to camera coordinates
#         pts_3d_cam = torch.bmm(torch.inverse(K0).unsqueeze(0).repeat(B, 1, 1),
#                                pts_3d_cam.transpose(1, 2)).transpose(1, 2)
        
#         # Transform to other camera
#         R = T_0to1[:, :3, :3]  # (B, 3, 3)
#         t = T_0to1[:, :3, 3:4]  # (B, 3, 1)
#         pts_3d_transformed = torch.bmm(R, pts_3d_cam.transpose(1, 2)) + t  # (B, 3, H*W)
#         pts_3d_transformed = pts_3d_transformed.transpose(1, 2)  # (B, H*W, 3)
        
#         # Project to image plane
#         pts_2d_projected = torch.bmm(K1.unsqueeze(0).repeat(B, 1, 1),
#                                      pts_3d_transformed.transpose(1, 2)).transpose(1, 2)
        
#         # Normalize by depth
#         z = pts_2d_projected[..., 2:3].clamp(min=1e-8)
#         pts_2d_normalized = pts_2d_projected[..., :2] / z  # (B, H*W, 2)
        
#         # Reshape back to spatial dimensions
#         pts_2d_normalized = pts_2d_normalized.reshape(B, H, W, 2)
        
#         # Compute flow (in downscaled coordinates)
#         flow_gt = (pts_2d_normalized - grid_scaled) / scale  # Scale back to feature resolution
        
#         # Compute valid mask
#         valid_proj = (pts_2d_normalized[..., 0] >= 0) & \
#                     (pts_2d_normalized[..., 0] < W * scale) & \
#                     (pts_2d_normalized[..., 1] >= 0) & \
#                     (pts_2d_normalized[..., 1] < H * scale)
        
#         valid_mask = valid_depth & valid_proj
        
#         return flow_gt, valid_mask
    
#     def compute_geometric_consistency_loss(self, data):
#         """
#         Geometric consistency loss for the geometry head.
        
#         Ensures that geometric features maintain epipolar constraints.
#         """
#         if 'feat_geom_0' not in data or 'feat_geom_1' not in data:
#             return torch.tensor(0.0, device=data['feat_8_0'].device)
        
#         feat_geom_0 = data['feat_geom_0']  # (B, d_geom, H*W)
#         feat_geom_1 = data['feat_geom_1']
        
#         B = data['bs']
        
#         # Reshape to spatial dimensions
#         H, W = data['h_8'], data['w_8']
#         feat_geom_0 = feat_geom_0.view(B, -1, H, W)
#         feat_geom_1 = feat_geom_1.view(B, -1, H, W)
        
#         # Compute similarity matrix using geometric features
#         feat_geom_0_norm = F.normalize(feat_geom_0, p=2, dim=1)
#         feat_geom_1_norm = F.normalize(feat_geom_1, p=2, dim=1)
        
#         # Flatten for similarity computation
#         feat_geom_0_flat = rearrange(feat_geom_0_norm, 'b c h w -> b (h w) c')
#         feat_geom_1_flat = rearrange(feat_geom_1_norm, 'b c h w -> b (h w) c')
        
#         # Compute similarity matrix
#         sim_matrix = torch.bmm(feat_geom_0_flat, feat_geom_1_flat.transpose(1, 2))  # (B, HW, HW)
        
#         # Apply temperature scaling
#         temperature = 0.1
#         sim_matrix = sim_matrix / temperature
        
#         # Get ground truth correspondences from conf_matrix_gt if available
#         if 'conf_matrix_gt' in data:
#             gt_matrix = data['conf_matrix_gt']  # (B, HW, HW)
            
#             # Compute cross-entropy loss
#             # Positive pairs (matches)
#             pos_mask = gt_matrix > 0
#             if pos_mask.any():
#                 pos_sim = sim_matrix[pos_mask]
#                 # Negative log-likelihood for positive pairs
#                 pos_loss = -F.logsigmoid(pos_sim).mean()
#             else:
#                 pos_loss = torch.tensor(0.0, device=sim_matrix.device)
            
#             # Negative pairs (non-matches)
#             neg_mask = gt_matrix == 0
#             # Sample negative pairs to avoid imbalance
#             num_neg_samples = min(pos_mask.sum() * 5, neg_mask.sum())
#             if num_neg_samples > 0 and neg_mask.any():
#                 neg_sim_all = sim_matrix[neg_mask]
#                 # Random sampling of negatives
#                 perm = torch.randperm(neg_sim_all.shape[0])[:num_neg_samples]
#                 neg_sim = neg_sim_all[perm]
#                 # Negative log-likelihood for negative pairs
#                 neg_loss = -F.logsigmoid(-neg_sim).mean()
#             else:
#                 neg_loss = torch.tensor(0.0, device=sim_matrix.device)
            
#             geom_consistency_loss = (pos_loss + neg_loss) / 2
#         else:
#             # If no GT available, use self-consistency
#             # Enforce that geometric features are distinctive
#             geom_consistency_loss = self.compute_distinctiveness_loss(sim_matrix)
        
#         return geom_consistency_loss
    
#     def compute_distinctiveness_loss(self, sim_matrix):
#         """
#         Encourage geometric features to be distinctive.
#         Each point should have high similarity with few other points.
#         """
#         B, N, M = sim_matrix.shape
        
#         # Compute entropy of similarity distributions
#         sim_probs_0to1 = F.softmax(sim_matrix, dim=2)
#         sim_probs_1to0 = F.softmax(sim_matrix, dim=1)
        
#         # Entropy (we want low entropy = peaky distribution)
#         entropy_0to1 = -(sim_probs_0to1 * torch.log(sim_probs_0to1 + 1e-8)).sum(dim=2).mean()
#         entropy_1to0 = -(sim_probs_1to0 * torch.log(sim_probs_1to0 + 1e-8)).sum(dim=1).mean()
        
#         # Negative entropy as loss (minimize entropy)
#         return (entropy_0to1 + entropy_1to0) / 2
    
#     def compute_sub_pixel_loss(self, data):
#         """
#         Sub-pixel refinement loss (inherited from JamMa).
#         """
#         if 'mkpts0_f_train' not in data or 'mkpts1_f_train' not in data:
#             return torch.tensor(0.0, device=data['feat_8_0'].device)
        
#         # Compute essential matrix
#         Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
#         E_mat = Tx @ data['T_0to1'][:, :3, :3]
        
#         m_bids = data['m_bids']
#         pts0 = data['mkpts0_f_train']
#         pts1 = data['mkpts1_f_train']
        
#         # Symmetric epipolar distance
#         sym_dist = self._symmetric_epipolar_distance(
#             pts0, pts1, E_mat[m_bids], data['K0'][m_bids], data['K1'][m_bids]
#         )
        
#         # Filter high-error matches
#         if len(sym_dist) == 0:
#             return torch.tensor(0.0, device=data['feat_8_0'].device)
        
#         loss = sym_dist[sym_dist < 1e-4]
#         if len(loss) == 0:
#             loss = sym_dist * 1e-9
        
#         return loss.mean()
    
#     def _symmetric_epipolar_distance(self, pts0, pts1, E, K0, K1):
#         """Compute symmetric epipolar distance."""
#         # Normalize points
#         pts0 = (pts0 - K0[:, [0, 1], [2, 2]]) / K0[:, [0, 1], [0, 1]]
#         pts1 = (pts1 - K1[:, [0, 1], [2, 2]]) / K1[:, [0, 1], [0, 1]]
#         pts0 = convert_points_to_homogeneous(pts0)
#         pts1 = convert_points_to_homogeneous(pts1)

#         # Compute epipolar lines
#         Ep0 = (pts0[:,None,:] @ E.transpose(-2,-1)).squeeze(1)
#         p1Ep0 = torch.sum(pts1 * Ep0, -1)
#         Etp1 = (pts1[:,None,:] @ E).squeeze(1)

#         # Symmetric distance
#         d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2 + 1e-9) + 
#                        1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2 + 1e-9))
#         return d
    
#     @torch.no_grad()
#     def compute_c_weight(self, data):
#         """Compute element-wise weights for coarse loss."""
#         if 'mask0' in data:
#             c_weight = (data['mask0'].flatten(-2)[..., None] * 
#                        data['mask1'].flatten(-2)[:, None]).float()
#         else:
#             c_weight = None
#         return c_weight
    
#     def forward(self, data):
#         """
#         Compute total loss for AS-Mamba.
        
#         Args:
#             data: Dictionary containing model outputs and ground truth
            
#         Updates:
#             data['loss']: Total loss
#             data['loss_scalars']: Individual loss components for logging
#         """
#         loss_scalars = {}
        
#         # 1. Coarse-level matching loss
#         c_weight = self.compute_c_weight(data)
#         loss_c = self.compute_coarse_loss(data, weight=c_weight)
#         loss_c *= self.coarse_weight
#         loss = loss_c
#         loss_scalars['loss_c'] = loss_c.clone().detach().cpu()
        
#         # 2. Fine-level matching loss
#         if 'conf_matrix_fine' in data:
#             loss_f = self.compute_fine_matching_loss(data)
#             loss_f *= self.fine_weight
#             loss = loss + loss_f
#             loss_scalars['loss_f'] = loss_f.clone().detach().cpu()
#         else:
#             loss_scalars['loss_f'] = torch.tensor(0.0)
        
#         # 3. Flow prediction loss (NEW)
#         loss_flow = self.compute_flow_loss(data)
#         if loss_flow is not None and loss_flow > 0:
#             loss_flow *= self.flow_weight
#             loss = loss + loss_flow
#             loss_scalars['loss_flow'] = loss_flow.clone().detach().cpu()
#         else:
#             loss_scalars['loss_flow'] = torch.tensor(0.0)
        
#         # 4. Geometric consistency loss (NEW)
#         loss_geom = self.compute_geometric_consistency_loss(data)
#         if loss_geom is not None and loss_geom > 0:
#             loss_geom *= self.geom_weight
#             loss = loss + loss_geom
#             loss_scalars['loss_geom'] = loss_geom.clone().detach().cpu()
#         else:
#             loss_scalars['loss_geom'] = torch.tensor(0.0)
        
#         # 5. Sub-pixel refinement loss
#         loss_sub = self.compute_sub_pixel_loss(data)
#         if loss_sub is not None and loss_sub > 0:
#             loss_sub *= self.sub_weight
#             loss = loss + loss_sub
#             loss_scalars['loss_sub'] = loss_sub.clone().detach().cpu()
#         else:
#             loss_scalars['loss_sub'] = torch.tensor(0.0)
        
#         # Update data with losses
#         loss_scalars['loss'] = loss.clone().detach().cpu()
#         data.update({
#             'loss': loss,
#             'loss_scalars': loss_scalars
#         })
        
#         return loss


# def create_meshgrid(H, W, device='cpu'):
#     """Create a meshgrid of pixel coordinates."""
#     y, x = torch.meshgrid(
#         torch.arange(H, device=device, dtype=torch.float32),
#         torch.arange(W, device=device, dtype=torch.float32),
#         indexing='ij'
#     )
#     grid = torch.stack([x, y], dim=-1)  # (H, W, 2)
#     return grid