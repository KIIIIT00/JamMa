import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch.utils.checkpoint import checkpoint
from src.jamma.utils.utils import KeypointEncoder_wo_score, up_conv4, MLPMixerEncoderLayer, normalize_keypoints
from src.jamma.mamba_module import JointMambaMultiHead
from src.jamma.as_mamba_block import AS_Mamba_Block
from src.jamma.matching_module import CoarseMatching, FineSubMatching
from src.utils.profiler import PassThroughProfiler
torch.backends.cudnn.deterministic = True
INF = 1E9


class AS_Mamba(nn.Module):
    """
    AS-Mamba: Adaptive Span Mamba for Image Matching
    
    This model combines the efficiency of JamMa's Mamba-based architecture
    with ASpanFormer's adaptive span mechanism for state-of-the-art image matching.
    
    Architecture:
    1. Feature Encoder (CNN backbone)
    2. Mamba Initializer (Multi-Head)
    3. N x AS-Mamba Blocks (Adaptive processing)
    4. Matching Module (Coarse + Fine)
    """
    def __init__(self, config, profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        
        # Extract key dimensions from config
        self.d_model_c = self.config['coarse']['d_model']  # 256
        self.d_model_f = self.config['fine']['d_model']    # 128 or 64
        
        # AS-Mamba specific configurations
        self.n_blocks = config.get('as_mamba', {}).get('n_blocks', 3)  # Number of AS-Mamba blocks
        self.d_geom = config.get('as_mamba', {}).get('d_geom', 64)     # Geometric feature dimension
        self.use_kan_flow = config.get('as_mamba', {}).get('use_kan_flow', False)  # Use KAN for flow prediction
        self.global_depth = config.get('as_mamba', {}).get('global_depth', 4)  # Depth for global Mamba
        self.local_depth = config.get('as_mamba', {}).get('local_depth', 4)    # Depth for local Mamba
        
        self.coarse_adapter = nn.Conv2d(256, self.d_model_c, kernel_size=1)

        # Keypoint encoder for position encoding
        # TODO: coarse.d_modelの値によって変化
        # self.kenc = KeypointEncoder_wo_score(self.d_model_c, [32, 64, 128, self.d_model_c])
        self.kenc = KeypointEncoder_wo_score(self.d_model_c, [128, self.d_model_c])
        
        # Mamba Initializer - Multi-Head version for initial global feature interaction
        self.mamba_initializer = JointMambaMultiHead(
            feature_dim=self.d_model_c, 
            depth=4,  # Initial shallow processing
            d_geom=self.d_geom,
            return_geometry=True,  # Enable geometry head output
            rms_norm=True, 
            residual_in_fp32=True, 
            fused_add_norm=True, 
            profiler=self.profiler
        )
        
        # Stack of AS-Mamba Blocks for iterative refinement
        self.as_mamba_blocks = nn.ModuleList([
            AS_Mamba_Block(
                d_model=self.d_model_c,
                d_geom=self.d_geom,
                d_ffn=self.d_model_c * 2,
                global_depth=self.global_depth,
                local_depth=self.local_depth,
                dropout=0.1,
                use_kan_flow=self.use_kan_flow
            )
            for _ in range(self.n_blocks)
        ])
        
        # Coarse-level matching module
        self.coarse_matching = CoarseMatching(config['match_coarse'], self.profiler)

        # Feature pyramid network for multi-scale processing
        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2*dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)

        # Fine-level encoder and matching
        W = self.config['fine_window_size']
        self.fine_enc = nn.ModuleList([MLPMixerEncoderLayer(2*W**2, 64) for _ in range(4)])
        self.fine_matching = FineSubMatching(config, self.profiler)
        
        # Optional: Geometry-aware fine matching (future extension)
        self.use_geom_for_fine = config.get('as_mamba', {}).get('use_geom_for_fine', False)

    def coarse_match(self, data):
        """
        Perform coarse-level matching with AS-Mamba blocks.
        
        This replaces the original JamMa's simple JointMamba with our
        sophisticated multi-block adaptive processing.
        """
        # Extract and prepare features
        desc0, desc1 = data['feat_8_0'], data['feat_8_1']
        
        desc0 = self.coarse_adapter(desc0)
        desc1 = self.coarse_adapter(desc1)

        desc0, desc1 = desc0.flatten(2, 3), desc1.flatten(2, 3)
        # desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        
        # Keypoint normalization
        kpts0 = normalize_keypoints(kpts0, data['imagec_0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['imagec_1'].shape[-2:])

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        
        # Update data with position-encoded features
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        with self.profiler.profile("mamba initializer"):
            # Initial global feature interaction with Multi-Head Mamba
            self.mamba_initializer(data)
            # Now data contains both feat_8_0/1 (matching) and feat_geom_0/1 (geometry)

        with self.profiler.profile("as-mamba blocks"):
            # Iterative refinement through AS-Mamba blocks
            for block_idx, as_block in enumerate(self.as_mamba_blocks):
                with self.profiler.profile(f"as-mamba block {block_idx}"):
                    data = as_block(data)
                    # Each block updates feat_8_0/1, feat_geom_0/1, and adds flow_map

        # Prepare for matching
        mask_c0 = mask_c1 = None
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        with self.profiler.profile("coarse matching"):
            # Use refined features for matching
            self.coarse_matching(
                data['feat_8_0'].transpose(1,2), 
                data['feat_8_1'].transpose(1,2), 
                data, 
                mask_c0=mask_c0, 
                mask_c1=mask_c1
            )

    def inter_fpn(self, feat_8, feat_4):
        """Feature pyramid network for multi-scale feature propagation."""
        d2 = self.up2(feat_8)  # 1/4
        d2 = self.act(self.conv7a(torch.cat([feat_4, d2], 1)))
        feat_4 = self.act(self.conv7b(d2))

        d1 = self.up3(feat_4)  # 1/2
        d1 = self.act(self.conv8a(d1))
        feat_2 = self.conv8b(d1)
        return feat_2

    def fine_preprocess(self, data, profiler):
        """
        Preprocess features for fine-level matching.
        
        Optionally incorporates geometric features for better fine matching.
        """
        data['resolution1'] = 8
        stride = data['resolution1'] // self.config['resolution'][1]
        W = self.config['fine_window_size']
        
        # Prepare features
        feat_8 = torch.cat([data['feat_8_0'], data['feat_8_1']], 0).view(
            2*data['bs'], data['c'], data['h_8'], -1
        )
        feat_4 = torch.cat([data['feat_4_0'], data['feat_4_1']], 0)

        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            feat1 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            return feat0, feat1

        # Multi-scale feature propagation
        feat_f = self.inter_fpn(feat_8, feat_4)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        data.update({'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})

        # Unfold (crop) all local windows
        pad = 0 if W % 2 == 0 else W//2
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)

        # Select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # Optional: Incorporate geometric features for fine matching
        if self.use_geom_for_fine and 'feat_geom_0' in data:
            # This would require additional processing to incorporate geometric cues
            # For now, we proceed with standard fine matching
            pass

        # Fine-level feature encoding
        feat_f = torch.cat([feat_f0_unfold, feat_f1_unfold], 1).transpose(1, 2)
        for layer in self.fine_enc:
            feat_f = layer(feat_f)
        feat_f0_unfold, feat_f1_unfold = feat_f[:, :, :W**2], feat_f[:, :, W**2:]
        
        return feat_f0_unfold, feat_f1_unfold

    def forward(self, data, mode='test'):
        """
        Forward pass of AS-Mamba.
        
        Args:
            data: Dictionary containing image features and metadata
            mode: 'train', 'val', or 'test'
            
        Returns:
            Updated data dictionary with matching results
        """
        self.mode = mode
        
        # Store input dimensions
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })

        # Coarse-level matching with AS-Mamba blocks
        self.coarse_match(data)

        with self.profiler.profile("fine matching"):
            # Fine-level preprocessing
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data, self.profiler)

            # Fine-level matching and sub-pixel refinement
            self.fine_matching(
                feat_f0_unfold.transpose(1, 2), 
                feat_f1_unfold.transpose(1, 2), 
                data
            )
        
        # Store AS-Mamba specific outputs for analysis
        if mode in ['val', 'test'] and 'flow_map' in data:
            # Keep flow predictions for visualization
            data['as_mamba_flow'] = data.get('flow_map')
            data['as_mamba_spans'] = data.get('adaptive_spans')
        
        return data