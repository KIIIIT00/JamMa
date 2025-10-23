from math import log
from loguru import logger

import torch
import torch.nn.functional as F
from einops import repeat
from kornia.utils import create_meshgrid
from einops.einops import rearrange
from .geometry import warp_kpts, warp_kpts_fine


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['imagec_0'].device
    N, _, H0, W0 = data['imagec_0'].shape
    _, _, H1, W1 = data['imagec_1'].shape
    scale = config['AS_MAMBA']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    valid_mask0, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    valid_mask1, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_i[~valid_mask0] = 0
    w_pt1_i[~valid_mask1] = 0
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    arange_1 = torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    arange_0 = torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    arange_1[nearest_index1 == 0] = 0
    arange_0[nearest_index0 == 0] = 0
    arange_b = torch.arange(N, device=device).unsqueeze(1)

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    conf_matrix_gt[arange_b, arange_1, nearest_index1] = 1
    conf_matrix_gt[arange_b, nearest_index0, arange_0] = 1
    conf_matrix_gt[:, 0, 0] = False

    b_ids, i_ids, j_ids = conf_matrix_gt.nonzero(as_tuple=True)

    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'num_candidates_max': b_ids.shape[0],
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
    

@torch.no_grad()
def spvs_fine(data, config):
    """
    Args:
        data (dict): {
            'b_ids': [M]
            'i_ids': [M]
            'j_ids': [M]
        }
        
    Update:
        data (dict): {
            conf_matrix_f_gt: [N, W_f^2, W_f^2], in original image resolution
            }

    """
    # 1. misc
    device = data['imagec_0'].device
    N, _, H0, W0 = data['imagec_0'].shape
    _, _, H1, W1 = data['imagec_1'].shape
    scale = config['AS_MAMBA']['RESOLUTION'][1]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    scale_f_c = config['AS_MAMBA']['RESOLUTION'][0] // config['AS_MAMBA']['RESOLUTION'][1]
    W_f = config['AS_MAMBA']['FINE_WINDOW_SIZE']
    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids_fine'], data['i_ids_fine'], data['j_ids_fine']

    if len(b_ids) == 0:
        data.update({"conf_matrix_f_gt": torch.zeros(1,W_f*W_f,W_f*W_f, device=device)})
        return

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).repeat(N, 1, 1, 1)
    grid_pt0_i = scale0[:,None,...] * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).repeat(N, 1, 1, 1)
    grid_pt1_i = scale1[:,None,...] * grid_pt1_c
    
    # unfold (crop windows) all local windows
    stride_f = data['hw0_f'][0] // data['hw0_c'][0]

    pad = 0 if W_f % 2 == 0 else W_f // 2
    grid_pt0_i = rearrange(grid_pt0_i, 'n h w c -> n c h w')
    grid_pt0_i = F.unfold(grid_pt0_i, kernel_size=(W_f, W_f), stride=stride_f, padding=pad)
    grid_pt0_i = rearrange(grid_pt0_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt0_i = grid_pt0_i[b_ids, i_ids]

    grid_pt1_i = rearrange(grid_pt1_i, 'n h w c -> n c h w')
    grid_pt1_i = F.unfold(grid_pt1_i, kernel_size=(W_f, W_f), stride=stride_f, padding=pad)
    grid_pt1_i = rearrange(grid_pt1_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt1_i = grid_pt1_i[b_ids, j_ids]

    # warp kpts bi-directionally and resize them to fine-level resolution
    # (no depth consistency check
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts_fine(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'], b_ids)
    _, w_pt1_i = warp_kpts_fine(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'], b_ids)
    w_pt0_f = w_pt0_i / scale1[b_ids]
    w_pt1_f = w_pt1_i / scale0[b_ids]

    mkpts0_c_scaled_to_f = torch.stack(
        [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
        dim=1) * scale_f_c - pad
    mkpts1_c_scaled_to_f = torch.stack(
        [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
        dim=1) * scale_f_c - pad
    
    w_pt0_f = w_pt0_f - mkpts1_c_scaled_to_f[:,None,:]
    w_pt1_f = w_pt1_f - mkpts0_c_scaled_to_f[:,None,:]

    # 3. check if mutual nearest neighbor
    w_pt0_f_round = w_pt0_f[:, :, :].round().long()
    w_pt1_f_round = w_pt1_f[:, :, :].round().long()
    M = w_pt0_f.shape[0]

    nearest_index1 = w_pt0_f_round[..., 0] + w_pt0_f_round[..., 1] * W_f
    nearest_index0 = w_pt1_f_round[..., 0] + w_pt1_f_round[..., 1] * W_f

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_f_round, W_f, W_f)] = 0
    nearest_index0[out_bound_mask(w_pt1_f_round, W_f, W_f)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(W_f*W_f, device=device)[None].repeat(M, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_f_gt = torch.zeros(M, W_f*W_f, W_f*W_f, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_f_gt[b_ids, i_ids, j_ids] = 1

    data.update({"conf_matrix_f_gt": conf_matrix_f_gt})

def compute_supervision_flow(data, config):
    """
    Compute ground truth flow for AS-Mamba supervision.
    
    This is CRITICAL for AS-Mamba's adaptive span mechanism.
    Flow supervision enables the model to learn:
    1. Accurate correspondence prediction
    2. Uncertainty estimation
    3. Adaptive window size selection
    
    Args:
        data: Batch dictionary containing:
            - conf_matrix_gt: Ground truth correspondence matrix (B, H0*W0, H1*W1)
            - hw0_c, hw1_c: Coarse feature map dimensions
        config: Configuration dictionary
    
    Updates data with:
        - flow_gt_0to1: Ground truth flow from image 0 to 1 (B, H0, W0, 2)
        - flow_gt_1to0: Ground truth flow from image 1 to 0 (B, H1, W1, 2)
        - spv_b_ids, spv_i_ids, spv_j_ids: Supervision indices for sparse supervision
    """
    
    # Extract dimensions
    bs = data['conf_matrix_gt'].shape[0]
    h0, w0 = data['hw0_c']
    h1, w1 = data['hw1_c']
    # print(f"DEBUG data hw0_c: {data['hw0_c']}")
    # print(f"DEBUG data hw1_c: {data['hw1_c']}")
    # print(f"DEBUG: h0 h1: {h0}, {h1}")
    
    device = data['conf_matrix_gt'].device
    
    # Get ground truth correspondences
    conf_gt = data['conf_matrix_gt']  # (B, H0*W0, H1*W1)
    
    # Find positive matches for sparse supervision
    # This gives us the indices where ground truth matches exist
    spv_b_ids = []  # Batch indices
    spv_i_ids = []  # Source indices (in image 0)
    spv_j_ids = []  # Target indices (in image 1)
    
    for b in range(bs):
        pos_indices = torch.nonzero(conf_gt[b] == 1, as_tuple=False)
        if len(pos_indices) > 0:
            i_ids = pos_indices[:, 0]  # Source positions
            j_ids = pos_indices[:, 1]  # Target positions
            
            spv_b_ids.append(torch.full((len(i_ids),), b, device=device))
            spv_i_ids.append(i_ids)
            spv_j_ids.append(j_ids)
    
    # Concatenate all supervision indices
    if len(spv_b_ids) > 0:
        spv_b_ids = torch.cat(spv_b_ids, dim=0)
        spv_i_ids = torch.cat(spv_i_ids, dim=0)
        spv_j_ids = torch.cat(spv_j_ids, dim=0)
    else:
        # No ground truth matches - create dummy supervision
        spv_b_ids = torch.zeros(1, dtype=torch.long, device=device)
        spv_i_ids = torch.zeros(1, dtype=torch.long, device=device)
        spv_j_ids = torch.zeros(1, dtype=torch.long, device=device)
    
    # Compute flow ground truth
    # Flow is the displacement from source position to target position
    # print(f"DEBUG: type bs: {type(bs)}")
    # print(f"DEBUG: type h0: {type(h0)}")
    # print(f"DEBUG: h0: {h0}")
    # print(f"DEBUG: type w1: {type(w1)}")
    # print(f"DEBUG: w1: {w1}")

    h0 = h0[0].long()
    w0 = w0[0].long()
    h1 = h1[0].long()
    w1 = w1[0].long()

    # print(f"DEBUG: type bs: {type(bs)}")
    # print(f"DEBUG: type h0: {type(h0)}")
    # print(f"DEBUG: h0: {h0}")
    # print(f"DEBUG: type h1: {type(h1)}")
    # print(f"DEBUG h1: {h1}")
    # print(f"DEBUG: type w0: {type(w0)}")
    # print(f"DEBUG: w0: {w0}")
    # print(f"DEBUG: type w1: {type(w1)}")
    # print(f"DEBUG w1: {w1}")
    flow_gt_0to1 = torch.zeros(bs, h0, w0, 2, device=device, dtype=torch.float32)
    flow_gt_1to0 = torch.zeros(bs, h1, w1, 2, device=device, dtype=torch.float32)
    
    for idx in range(len(spv_b_ids)):
        b = spv_b_ids[idx]
        i = spv_i_ids[idx]
        j = spv_j_ids[idx]
        if i >= h0 * w0:
            continue
        
        # Convert linear indices to 2D coordinates
        i_y, i_x = i // w0, i % w0
        
        if j >= h1 * w1:
            continue
        j_y, j_x = j // w1, j % w1
        
        # Flow from i to j (0 -> 1)
        flow_x = j_x.float() - i_x.float()
        flow_y = j_y.float() - i_y.float()
        # print(f"flow_x: {flow_x}, flow_y: {flow_y}")
        flow_gt_0to1[b, i_y, i_x, 0] = flow_x
        flow_gt_0to1[b, i_y, i_x, 1] = flow_y
        
        # Flow from j to i (1 -> 0)
        flow_gt_1to0[b, j_y, j_x, 0] = -flow_x
        flow_gt_1to0[b, j_y, j_x, 1] = -flow_y
    
    # Update data dictionary
    data.update({
        'flow_gt_0to1': flow_gt_0to1,
        'flow_gt_1to0': flow_gt_1to0,
        'spv_b_ids': spv_b_ids,
        'spv_i_ids': spv_i_ids,
        'spv_j_ids': spv_j_ids
    })
    
    return data


def compute_optimal_spans(flow_magnitude, uncertainty, config):
    """
    Compute optimal adaptive spans for analysis/debugging.
    
    This can be used to:
    1. Validate learned span predictions
    2. Analyze span distribution
    3. Debug adaptive span mechanism
    
    Args:
        flow_magnitude: Magnitude of flow vectors (B, H, W)
        uncertainty: Flow uncertainty (B, H, W)
        config: Configuration
    
    Returns:
        optimal_spans: Recommended span sizes (B, H, W)
    """
    base_span = config['asmamba'].get('base_span', 7)
    max_span = config['asmamba'].get('max_span', 15)
    min_span = config['asmamba'].get('min_span', 3)
    
    # Higher flow magnitude -> larger span needed
    # Higher uncertainty -> larger span needed (to capture more context)
    span_adjustment = torch.clamp(
        flow_magnitude * 0.5 + uncertainty * 2.0,
        min=0.0,
        max=float(max_span - base_span)
    )
    
    optimal_spans = torch.clamp(
        base_span + span_adjustment,
        min=min_span,
        max=max_span
    ).long()
    
    return optimal_spans


def compute_flow_errors(predicted_flow, gt_flow, mask=None):
    """
    Compute flow prediction errors for analysis.
    
    Args:
        predicted_flow: Predicted flow (B, H, W, 4) with [dx, dy, ux, uy]
        gt_flow: Ground truth flow (B, H, W, 2)
        mask: Valid region mask (B, H, W)
    
    Returns:
        Dictionary with error statistics
    """
    # Extract predicted flow (first 2 channels)
    pred_flow_xy = predicted_flow[..., :2]
    
    # Compute endpoint error (EPE)
    epe = torch.sqrt(((pred_flow_xy - gt_flow) ** 2).sum(-1))
    
    if mask is not None:
        epe = epe[mask]
    
    errors = {
        'mean_epe': epe.mean().item(),
        'median_epe': epe.median().item(),
        'max_epe': epe.max().item(),
        '1px_accuracy': (epe < 1.0).float().mean().item(),
        '3px_accuracy': (epe < 3.0).float().mean().item(),
    }
    
    return errors


def compute_adaptive_span_statistics(spans_x, spans_y):
    """
    Compute statistics about adaptive span distribution.
    
    Useful for:
    1. Monitoring span learning progress
    2. Identifying problematic regions
    3. Validating span diversity
    
    Args:
        spans_x, spans_y: Adaptive spans (B, H, W)
    
    Returns:
        Dictionary with span statistics
    """
    stats = {
        'mean_span_x': spans_x.float().mean().item(),
        'mean_span_y': spans_y.float().mean().item(),
        'std_span_x': spans_x.float().std().item(),
        'std_span_y': spans_y.float().std().item(),
        'min_span': min(spans_x.min().item(), spans_y.min().item()),
        'max_span': max(spans_x.max().item(), spans_y.max().item()),
        'span_entropy': compute_entropy(torch.cat([spans_x.flatten(), spans_y.flatten()])),
    }
    
    return stats


def compute_entropy(values):
    """Compute entropy of discrete values (measure of diversity)."""
    unique, counts = torch.unique(values, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    return entropy.item()


def verify_flow_consistency(flow_0to1, flow_1to0, threshold=2.0):
    """
    Verify bidirectional flow consistency.
    
    For valid flows: flow_0to1(x) + flow_1to0(x + flow_0to1(x)) ≈ 0
    
    This can identify:
    1. Occlusions
    2. Prediction errors
    3. Non-rigid motion
    
    Args:
        flow_0to1: Flow from image 0 to 1 (B, H, W, 2)
        flow_1to0: Flow from image 1 to 0 (B, H, W, 2)
        threshold: Consistency threshold in pixels
    
    Returns:
        consistency_mask: Boolean mask of consistent flows
        consistency_error: Magnitude of inconsistency
    """
    B, H, W, _ = flow_0to1.shape
    device = flow_0to1.device
    
    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    coords = torch.stack([x_coords, y_coords], dim=-1).float()  # (H, W, 2)
    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    
    # Warp coordinates by flow_0to1
    warped_coords = coords + flow_0to1
    
    # Sample flow_1to0 at warped locations
    # Normalize coordinates to [-1, 1] for grid_sample
    warped_coords_norm = warped_coords.clone()
    warped_coords_norm[..., 0] = 2.0 * warped_coords_norm[..., 0] / (W - 1) - 1.0
    warped_coords_norm[..., 1] = 2.0 * warped_coords_norm[..., 1] / (H - 1) - 1.0
    
    # Reshape for grid_sample
    flow_1to0_perm = flow_1to0.permute(0, 3, 1, 2)  # (B, 2, H, W)
    warped_flow = F.grid_sample(
        flow_1to0_perm,
        warped_coords_norm,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    warped_flow = warped_flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
    
    # Compute consistency error
    consistency_error = torch.sqrt(((flow_0to1 + warped_flow) ** 2).sum(-1))
    consistency_mask = consistency_error < threshold
    
    return consistency_mask, consistency_error