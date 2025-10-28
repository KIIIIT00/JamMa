import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from loguru import logger
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
INF = 1e9


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def generate_random_mask(n, num_true):
    # 创建全 False 的掩码
    mask = torch.zeros(n, dtype=torch.bool)

    # 随机选择 num_true 个位置并设置为 True
    indices = torch.randperm(n)[:num_true]
    mask[indices] = True

    return mask


class CoarseMatching(nn.Module):
    def __init__(self, config, profiler):
        super().__init__()
        self.config = config
        # general config
        d_model = 256
        self.thr = config['thr']
        logger.debug(f"CoarseMatching threshold set to: {self.thr}")
        self.use_sm = config['use_sm']
        self.inference = config['inference']
        self.border_rm = config['border_rm']

        self.final_proj = nn.Linear(d_model, d_model, bias=True)

        self.temperature = config['dsmax_temperature']
        self.profiler = profiler

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                  feat_c1) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                torch.finfo(sim_matrix.dtype).min)
        if self.inference:
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_inference(sim_matrix, data))
        else:
            conf_matrix_0_to_1 = F.softmax(sim_matrix, 2)
            conf_matrix_1_to_0 = F.softmax(sim_matrix, 1)
            data.update({'conf_matrix_0_to_1': conf_matrix_0_to_1,
                         'conf_matrix_1_to_0': conf_matrix_1_to_0
                         })
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_training(conf_matrix_0_to_1, conf_matrix_1_to_0, data))

    @torch.no_grad()
    def get_coarse_match_training(self, conf_matrix_0_to_1, conf_matrix_1_to_0, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix_0_to_1.device

        # confidence thresholding
        # {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or(
            (conf_matrix_0_to_1 > self.thr) * (conf_matrix_0_to_1 == conf_matrix_0_to_1.max(dim=2, keepdim=True)[0]),
            (conf_matrix_1_to_0 > self.thr) * (conf_matrix_1_to_0 == conf_matrix_1_to_0.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = torch.maximum(conf_matrix_0_to_1[b_ids, i_ids, j_ids], conf_matrix_1_to_0[b_ids, i_ids, j_ids])
        # logger.debug(f"Number of coarse matches before sampling/padding: {len(b_ids)}")
        # logger.debug(f"mconf shape before sampling/padding: {mconf.shape}, requires_grad: {mconf.requires_grad}")
        # logger.debug(f"mconf sample before sampling/padding: {mconf[:10]}")

        # random sampling of training samples for fine-level XoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # the sampling is performed across all pairs in a batch without manually balancing
            # samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.config['train_coarse_percent'])
            num_matches_pred = len(b_ids)
            train_pad_num_gt_min = self.config['train_pad_num_gt_min']
            assert train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - train_pad_num_gt_min,),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data['spv_b_ids']),
                (max(num_matches_train - num_matches_pred,
                     train_pad_num_gt_min),),
                device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        # logger.debug(f"mkpts0_c shape: {mkpts0_c.shape}, requires_grad: {mkpts0_c.requires_grad}")
        # logger.debug(f"mkpts0_C[mconf != 0] shape: {mkpts0_c[mconf != 0].shape}, requires_grad: {mkpts0_c[mconf != 0].requires_grad}")
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mkpts0_c_train': mkpts0_c,
            'mkpts1_c_train': mkpts1_c,
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches

    @torch.no_grad()
    def get_coarse_match_inference(self, sim_matrix, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 2) if self.use_sm else sim_matrix

        # confidence thresholding and nearest neighbour for 0 to 1
        mask = (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=2, keepdim=True)[0])

        # unlike training, reuse the same conf martix to decrease the vram consumption
        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 1) if self.use_sm else sim_matrix

        # update mask {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or(mask,
                                (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = sim_matrix[b_ids, i_ids, j_ids]

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches are the current coarse level predictions
        coarse_matches.update({
            'mconf': mconf,
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
        })

        return coarse_matches


class FineSubMatching(nn.Module):
    """Fine-level and Sub-pixel matching"""

    def __init__(self, config, profiler):
        super().__init__()
        self.temperature = config['fine']['dsmax_temperature']
        self.W_f = config['fine_window_size']
        self.inference = config['fine']['inference']
        dim_f = 64
        self.fine_thr = config['fine']['thr']
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        self.subpixel_mlp = nn.Sequential(nn.Linear(2 * dim_f, 2 * dim_f, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(2 * dim_f, 4, bias=False))
        self.fine_spv_max = 500  # saving memory
        self.profiler = profiler

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):

        if not torch.isfinite(feat_f0_unfold).all() or not torch.isfinite(feat_f1_unfold).all():
                logger.error(f"[NaN DETECTED] Fine-level features (feat_f0_unfold or feat_f1_unfold) contain NaN/Inf!")
                logger.error(f"  - feat_f0_unfold has NaN: {torch.isnan(feat_f0_unfold).any()}")
                logger.error(f"  - feat_f1_unfold has NaN: {torch.isnan(feat_f1_unfold).any()}")
                logger.error(f"  - feat_f0_unfold has Inf: {torch.isinf(feat_f0_unfold).any()}")
                logger.error(f"  - feat_f1_unfold has Inf: {torch.isinf(feat_f1_unfold).any()}")
                
                # ここで意図的にクラッシュさせ、NaNの発生源（多くの場合 FineEnc_MLP）を特定する
                raise RuntimeError("NaN/Inf detected in fine-level features before matching.")
        
        M, WW, C = feat_f0_unfold.shape
        W_f = self.W_f
        assert WW == W_f * W_f, f"Expected window size {W_f*W_f}, but got {WW}"
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        
        # DEBUG
        global_step_str = data.get('global_step', 'N/A')
        # logger.debug(f"[Step {global_step_str}] FineSubMatching called with M={M}, WW={WW}, C={C}")
        

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            if self.inference:
                data.update({
                    'expec_f': torch.zeros(0, 3, device=feat_f0_unfold.device),
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                    'mconf_f': torch.zeros(0, device=feat_f0_unfold.device)
                })
            else:
                data.update({
                    'expec_f': torch.zeros(0, 3, device=feat_f0_unfold.device),
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                    'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
                    'mkpts0_f_train': data['mkpts0_c_train'],
                    'mkpts1_f_train': data['mkpts1_c_train'],
                    'conf_matrix_fine': torch.zeros(1, W_f * W_f, W_f * W_f, device=feat_f0_unfold.device),
                    'b_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                    'i_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                    'j_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                })
            # Add keys needed for training loss if M=0 occurs during training
            if self.training:
                 data.update({
                      'mkpts0_f_train': data.get('mkpts0_c_train', torch.empty(0, 2, device=feat_f0_unfold.device)),
                      'mkpts1_f_train': data.get('mkpts1_c_train', torch.empty(0, 2, device=feat_f0_unfold.device)),
                      # Add other training specific keys if needed
                 })
            return

        # 2025-10-25: feature extraction at the center of each window
        feat_f0_center = feat_f0_unfold[:, WW // 2, :]

        feat_f0_center = F.normalize(feat_f0_center, p=2, dim=-1)
        feat_f1 = F.normalize(feat_f1_unfold, p=2, dim=-1)
        
        # 2025-10-25: simirary
        sim_matrix = torch.einsum("mc,mwc->mw", feat_f0_center,feat_f1)
        
        # 2025-10-25: heatmap
        softmax_temp = self.temperature
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W_f, W_f)
        
        # 2025-10-25: DSNT
        coords_normalized = dsnt.spatial_expectation2d(heatmap.unsqueeze(1), True).squeeze(1)
        
        # 2025-10-25: 
        grid_normalized = create_meshgrid(W_f, W_f, True, heatmap.device).reshape(1, WW, 2) # (1, WW, 2)
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)
        
        expec_f = torch.cat([coords_normalized, std.unsqueeze(1)], -1)
        data.update({'expec_f': expec_f})
        # logger.debug(f"[Step {global_step_str}] Calculated expec_f shape: {expec_f.shape}, requires_grad: {expec_f.requires_grad}")
        
        # logger.debug(f"heatmap shape: {heatmap.shape}, requires_grad: {heatmap.requires_grad}")
        self.get_fine_match_compatible(coords_normalized, data, heatmap)
    
    @torch.no_grad()
    def get_fine_match_compatible(self, coords_normalized, data, heatmap):
        """Calculates final coordinates based on normalized coords,
           compatible with existing JamMa/AS-Mamba structure."""
        M = coords_normalized.shape[0]
        W_f = self.W_f
        scale = data['hw0_i'][0] / data['hw0_f'][0] # Image_res / Fine_feature_res

        # print("DEBUG: data['b_ids']:", data['b_ids'])
        # print("DEBUG: data['mkpts0_c'] shape:", data['mkpts0_c'].shape)
        # print("DEBUG: data[mkpts0_c] keys:", list(data.keys()))
        # mkpts0_f = data['mkpts0_c'][data['b_ids']] # (M, 2) 
        
        h0c, w0c = data['hw0_c']
        h1c, w1c = data['hw1_c']
        mkpts0_c_coords = torch.stack([data['i_ids'] % w0c, data['i_ids'] // w0c], dim=1).float()
        # logger.debug(f"mkpts0_c_coords shape: {mkpts0_c_coords.shape}, requires_grad: {mkpts0_c_coords.requires_grad}")
        mkpts1_c_coords = torch.stack([data['j_ids'] % w1c, data['j_ids'] // w1c], dim=1).float()
        # logger.debug(f"mkpts1_c_coords shape: {mkpts1_c_coords.shape}, requires_grad: {mkpts1_c_coords.requires_grad}")

        # mkpts1_f は coarse 座標に DSNT の結果を加味して補正
        # coords_normalized: [-1, 1] -> offset in fine feature pixel space: * (W_f / 2)
        # -> offset in image pixel space: * scale
        # offset = coords_normalized * (W_f / 2.0) * scale # (M, 2)

        # mkpts0_f
        scale_c_i_0 = data['hw0_i'][0] / data['hw0_c'][0]
        if 'scale0' in data and 'b_ids' in data:
            scale0_m = data['scale0'][data['b_ids']] # (M,)
            # logger.debug(f"scale0_m shape: {scale0_m.shape}, requires_grad: {scale0_m.requires_grad}")
            mkpts0_c_image_scale = (mkpts0_c_coords * scale_c_i_0) * scale0_m
            base_offset = coords_normalized * (W_f / 2.0) * scale
            offset = base_offset * scale0_m
        else:
            mkpts0_c_image_scale = mkpts0_c_coords * scale_c_i_0
            offset = coords_normalized * (W_f / 2.0) * scale # (M, 2)
        
        mkpts0_f = mkpts0_c_image_scale + offset # (M, 2)
        # logger.debug(f"mkpts0_f shape[get_fine_match_compatible]: {mkpts0_f.shape}, requires_grad: {mkpts0_f.requires_grad}")
        
        scale_c_i_1 = data['hw1_i'][0] / data['hw1_c'][0] # Image_res / Coarse_feature_res
        # スケーリングファクタ (もしあれば適用)
        if 'scale1' in data and 'b_ids' in data:
            # Ensure scale1 corresponds to the M matches
            scale1_m = data['scale1'][data['b_ids']] # (M,)
            # offset = offset * scale1_m.unsqueeze(-1)
            # logger.debug(f"scale1_m shape: {scale1_m.shape}, requires_grad: {scale1_m.requires_grad}")
            mkpts1_c_image_scale = (mkpts1_c_coords * scale_c_i_1) * scale1_m
            base_offset = coords_normalized * (W_f / 2.0) * scale
            
            offset = base_offset * scale1_m
        
        else:
            mkpts1_c_image_scale = mkpts1_c_coords * scale_c_i_1
            offset = coords_normalized * (W_f / 2.0) * scale # (M, 2)

        mkpts1_f = mkpts1_c_image_scale + offset # (M, 2)

        # logger.debug(f"mkpts1_f shape[get_fine_match_compatible]: {mkpts1_f.shape}, requires_grad: {mkpts1_f.requires_grad}")

        # mconf (信頼度) を計算 (オプション、評価用)
        mconf, _ = heatmap.view(M, -1).max(dim=1)

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "mconf_f": mconf # 評価に必要な場合
        })

        # --- 学習用の座標 (mkpts0_f_train, mkpts1_f_train) ---
        # 元の JamMa では subpixel_mlp を使っていたが、ASpanFormer では使っていない
        # L_fine の勾配は expec_f を通じて流れるため、subpixel_mlp は不要かもしれない
        # ここでは、DSNTの結果を直接使った座標を学習用とする
        if self.training:
            mkpts0_f_train = mkpts0_f.detach().clone()
            mkpts1_f_train = mkpts1_f.detach().clone()

            update_dict = {
                'mkpts0_f_train': mkpts0_f_train,
                'mkpts1_f_train': mkpts1_f_train,
            }
            data.update({
                'mkpts0_f_train': mkpts0_f.detach().clone(), # 勾配はexpec_f経由なのでdetach
                'mkpts1_f_train': mkpts1_f.detach().clone()
                # 'conf_matrix_fine' など、他の学習に必要なキーもここかforwardの最後で追加
            })
        
        M = data['b_ids'].shape[0] 
        # logger.debug(f"Number of fine matches before sampling/padding: {M}")
        # logger.debug(f"mconf_f shape before sampling/padding: {mconf.shape}, requires_grad: {mconf.requires_grad}")
        train_mask = None
        if hasattr(self, 'fine_spv_max') and self.fine_spv_max is not None and self.fine_spv_max < M:
             try:
                  train_mask = generate_random_mask(M, self.fine_spv_max)
                  update_dict.update({
                       'b_ids_fine': data['b_ids'][train_mask],
                       'i_ids_fine': data['i_ids'][train_mask],
                       'j_ids_fine': data['j_ids'][train_mask],
                  })
             except NameError:
                  logger.error("'generate_random_mask' not defined. Using all matches.")
                  update_dict.update({
                      'b_ids_fine': data['b_ids'],
                      'i_ids_fine': data['i_ids'],
                      'j_ids_fine': data['j_ids'],
                  })
        else: # サンプリングしない場合
             update_dict.update({
                 'b_ids_fine': data['b_ids'],
                 'i_ids_fine': data['i_ids'],
                 'j_ids_fine': data['j_ids'],
             })

        data.update(update_dict)
        
        # feat_f0 = self.fine_proj(feat_f0_unfold)
        # feat_f1 = self.fine_proj(feat_f1_unfold)

    #     # normalize
    #     feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1] ** .5,
    #                            [feat_f0, feat_f1])
    #     sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0,
    #                               feat_f1) / self.temperature

    #     conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        
    #     # expec_f
    #     expec_f = self.calculate_expec_f(conf_matrix_fine, W_f)
    #     data.update({'expec_f': expec_f})
    #     logger.debug(f"[Step {global_step_str}] Calculated expec_f with shape: {expec_f.shape}, requires_grad: {expec_f.requires_grad}")
        

    #     # predict fine-level and sub-pixel matches from conf_matrix
    #     data.update(**self.get_fine_sub_match(conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data))
    
    # def calculate_expec_f(self, conf_matrix_fine, W_f):
    #     """Calculate the expected offset (expec_f) differentiably."""
    #     M, WW, _ = conf_matrix_fine.shape
    #     device = conf_matrix_fine.device

    #     # ウィンドウ内の相対座標グリッド [- (W-1)/2, ..., (W-1)/2] を作成
    #     center_offset = (W_f - 1) / 2.0
    #     grid_coords = torch.arange(W_f, device=device, dtype=torch.float32) - center_offset

    #     # グリッドを (WW, 2) に整形: [[x0,y0], [x1,y0], ..., [xW,yW]]
    #     grid_y, grid_x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
    #     grid = torch.stack([grid_x, grid_y], dim=-1).view(WW, 2) # (WW, 2)

    #     # 期待値計算 (Spatial Expectation)
    #     # conf_matrix_fine: (M, WW_0, WW_1)
        
    #     # E[pos1 | pos0] = sum_{pos1} ( P(pos1 | pos0) * pos1 )
    #     # P(pos1 | pos0) は softmax(sim[:, pos0, :], dim=1) に相当
    #     prob_map_1 = F.softmax(conf_matrix_fine.sum(dim=1) / self.temperature, dim=1) # (M, WW_1) 各点0に対する点1の確率分布
    #     # expec_1: 各点0に対する点1の期待座標 (M, 2)
    #     expec_1 = torch.einsum('mi,ic->mc', prob_map_1, grid)

    #     # E[pos0 | pos1] = sum_{pos0} ( P(pos0 | pos1) * pos0 )
    #     # P(pos0 | pos1) は softmax(sim[:, :, pos1], dim=1) に相当
    #     prob_map_0 = F.softmax(conf_matrix_fine.sum(dim=2) / self.temperature, dim=1) # (M, WW_0) 各点1に対する点0の確率分布
    #     # expec_0: 各点1に対する点0の期待座標 (M, 2)
    #     expec_0 = torch.einsum('mi,ic->mc', prob_map_0, grid)

    #     # expec_f は「点0の座標」から「点1の期待座標」へのオフセットとするのが一般的
    #     # 点0のウィンドウ中心はオフセット(0,0)に対応すると考える
    #     center_coords_0 = torch.zeros(M, 2, device=device)
    #     offset_pred = expec_1 - center_coords_0 # (M, 2)

    #     # 不確実性 (std) の計算 (オプション)
    #     # 例: 予測の分散やエントロピーから計算。ここでは仮に固定値とする
    #     std_dev = torch.ones(M, 1, device=device) * 0.1 # 仮の値

    #     expec_f = torch.cat([offset_pred, std_dev], dim=1) # (M, 3)
    #     return expec_f

    def get_fine_sub_match(self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data):
        with torch.no_grad():
            W_f = self.W_f

            # 1. confidence thresholding
            mask = conf_matrix_fine > self.fine_thr

            if mask.sum() == 0:
                mask[0, 0, 0] = 1
                conf_matrix_fine[0, 0, 0] = 1

            # match only the highest confidence
            mask = mask \
                   * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))

            # 3. find all valid fine matches
            # this only works when at most one `True` in each row
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

            # 4. update with matches in original image resolution

            # indices from coarse matches
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # scale (coarse level / fine-level)
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            # coarse level matches scaled to fine-level (1/2)
            mkpts0_c_scaled_to_f = torch.stack(
                [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            # updated b_ids after second thresholding
            updated_b_ids = b_ids_c[b_ids]

            # scales (image res / fine level)
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

            # fine-level discrete matches on window coordiantes
            mkpts0_f_window = torch.stack(
                [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')],
                dim=1)

            mkpts1_f_window = torch.stack(
                [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')],
                dim=1)

        # sub-pixel refinement
        sub_ref = self.subpixel_mlp(torch.cat([feat_f0_unfold[b_ids, i_ids], feat_f1_unfold[b_ids, j_ids]], dim=-1))
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=1)
        sub_ref0, sub_ref1 = sub_ref0.squeeze(1), sub_ref1.squeeze(1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        pad = 0 if W_f % 2 == 0 else W_f // 2
        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement)
        mkpts0_f1 = (mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - pad) * scale0  # + sub_ref0
        mkpts1_f1 = (mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - pad) * scale1  # + sub_ref1
        mkpts0_f_train = mkpts0_f1 + sub_ref0 * scale0  # + sub_ref0
        mkpts1_f_train = mkpts1_f1 + sub_ref1 * scale1  # + sub_ref1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # These matches is the current prediction (for visualization)
        sub_pixel_matches = {
            'm_bids': b_ids_c[b_ids[mconf != 0]],  # mconf == 0 => gt matches
            'mkpts0_f1': mkpts0_f1[mconf != 0],
            'mkpts1_f1': mkpts1_f1[mconf != 0],
            'mkpts0_f': mkpts0_f[mconf != 0],
            'mkpts1_f': mkpts1_f[mconf != 0],
            'mconf_f': mconf[mconf != 0]
        }

        # These matches are used for training
        # logger.debug(f"self.inference: {self.inference}")
        if not self.inference:
            logger.debug(f"Number of fine matches before sampling/padding: {len(data['b_ids'])}")
            if self.fine_spv_max is None or self.fine_spv_max > len(data['b_ids']):
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'],
                    'i_ids_fine': data['i_ids'],
                    'j_ids_fine': data['j_ids'],
                    'conf_matrix_fine': conf_matrix_fine
                })
            else:
                train_mask = generate_random_mask(len(data['b_ids']), self.fine_spv_max)
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'][train_mask],
                    'i_ids_fine': data['i_ids'][train_mask],
                    'j_ids_fine': data['j_ids'][train_mask],
                    'conf_matrix_fine': conf_matrix_fine[train_mask]
                })

        return sub_pixel_matches