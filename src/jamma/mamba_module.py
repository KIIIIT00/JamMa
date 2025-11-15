import torch
from torch import nn
import math
from typing import Tuple, Optional
from src.jamma.utils.utils import GLU_3
from mamba_ssm import Mamba
from functools import partial
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, LayerNorm
except ImportError:
    RMSNorm, LayerNorm = None, None
from src.utils.profiler import PassThroughProfiler


# class Block(nn.Module):
#     def __init__(
#             self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
#     ):
#         """
#         Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

#         This Block has a slightly different structure compared to a regular
#         prenorm Transformer block.
#         The standard block is: LN -> MHA/MLP -> Add.
#         [Ref: https://arxiv.org/abs/2002.04745]
#         Here we have: Add -> LN -> Mixer, returning both
#         the hidden_states (output of the mixer) and the residual.
#         This is purely for performance reasons, as we can fuse add and LayerNorm.
#         The residual needs to be provided (except for the very first block).
#         """
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.mixer = mixer_cls(dim)
#         self.norm = norm_cls(dim)
#         if self.fused_add_norm:
#             assert RMSNorm is not None, "RMSNorm import fails"
#             assert isinstance(
#                 self.norm, (nn.LayerNorm, RMSNorm)
#             ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

#     def forward(
#             self, desc, inference_params=None
#     ):
#         r"""Pass the input through the encoder layer.

#         Args:
#             hidden_states: the sequence to the encoder layer (required).
#             residual: hidden_states = Mixer(LN(residual))
#         """
#         hidden_states = self.norm(desc.to(dtype=self.norm.weight.dtype))
#         if self.residual_in_fp32:
#             desc = desc.to(torch.float32)
#         hidden_states = self.mixer(hidden_states, inference_params=inference_params)
#         return desc + hidden_states

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# def create_block(
#     d_model,
#     ssm_cfg=None,
#     norm_epsilon=1e-5,
#     drop_path=0.,
#     rms_norm=False,
#     residual_in_fp32=False,
#     fused_add_norm=False,
#     layer_idx=None,
#     device=None,
#     dtype=None,
#     if_bimamba=False,
#     bimamba_type="none",
#     if_devide_out=False,
#     init_layer_scale=None,
# ):
#     if if_bimamba:
#         bimamba_type = "v1"
#     if ssm_cfg is None:
#         ssm_cfg = {}
#     factory_kwargs = {"device": device, "dtype": dtype}
#     mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
#     norm_cls = partial(
#         nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
#     )
#     block = Block(
#         d_model,
#         mixer_cls,
#         norm_cls=norm_cls,
#         fused_add_norm=fused_add_norm,
#         residual_in_fp32=residual_in_fp32,
#     )
#     block.layer_idx = layer_idx
#     return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def scan_jego(desc0, desc1, step_size):
    desc_2w, desc_2h = torch.cat([desc0, desc1], 3), torch.cat([desc0, desc1], 2)
    _, _, org_h, org_2w = desc_2w.shape
    B, C, org_2h, org_w = desc_2h.shape

    H = org_h // step_size
    W = org_2w // step_size

    xs = desc_2w.new_empty((B, 4, C, H*W))

    xs[:, 0] = desc_2w[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 1] = desc_2h.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)  # [w/2, 2w/2]
    xs[:, 2] = desc_2w[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [h/2, 2w/2]
    xs[:, 3] = desc_2h.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [w/2, 2w/2]

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_jego(ys, ori_h: int, ori_w: int, step_size=2):
    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w = torch.zeros((B, C, new_h, 2*new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, new_h, 2*new_w))
    y_2h = torch.zeros((B, C, 2*new_h, new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, 2*new_h, new_w))

    y_2w[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, 1::step_size] = ys[:, 1].reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)
    y_2w[:, :, ::step_size, 1::step_size] = ys[:, 2].flip([2]).reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, ::step_size] = ys[:, 3].flip([2]).reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()
    desc0_2w, desc1_2w = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, desc1_2h = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w+desc0_2h, desc1_2w+desc1_2h


def scan_jego_seq(desc0, desc1, step_size):
    desc_2w, desc_2h = torch.cat([desc0, desc1], 3), torch.cat([desc0, desc1], 2)
    _, _, org_h, org_2w = desc_2w.shape
    B, C, org_2h, org_w = desc_2h.shape

    H = org_h // step_size
    W = org_2w // step_size

    xs = desc_2w.new_empty((B, 4, C, H*W))

    xs[:, 0] = desc_2h[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 1] = desc_2w.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)  # [w/2, 2w/2]
    xs[:, 2] = desc_2h[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [h/2, 2w/2]
    xs[:, 3] = desc_2w.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [w/2, 2w/2]

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_jego_seq(ys, ori_h: int, ori_w: int, step_size=2):
    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w = torch.zeros((B, C, new_h, 2*new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, new_h, 2*new_w))
    y_2h = torch.zeros((B, C, 2*new_h, new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, 2*new_h, new_w))

    y_2h[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, 2*H, W)
    y_2w[:, :, 1::step_size, 1::step_size] = ys[:, 1].reshape(B, C, 2*W, H).transpose(dim0=2, dim1=3)
    y_2h[:, :, ::step_size, 1::step_size] = ys[:, 2].flip([2]).reshape(B, C, 2*H, W)
    y_2w[:, :, 1::step_size, ::step_size] = ys[:, 3].flip([2]).reshape(B, C, 2*W, H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()
    desc0_2w, desc1_2w = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, desc1_2h = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w+desc0_2h, desc1_2w+desc1_2h


def scan_vim(desc0, desc1):
    B, C, org_h, org_w = desc0.shape
    desc_2w = torch.cat([desc0, desc1], 3)

    H = org_h
    W = desc_2w.shape[3]

    xs = desc_2w.new_empty((B, 2, C, H*W))

    xs[:, 0] = desc_2w.view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 1] = desc_2w.view(B, C, -1).flip([2])  # [w/2, 2w/2]

    xs = xs.view(B, 2, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_vim(ys, org_h, org_w):
    B, K, C, L = ys.shape

    y_2w_f = ys[:, 0].reshape(B, C, org_h, 2*org_w)
    y_2w_b = ys[:, 1].flip([2]).reshape(B, C, org_h, 2*org_w)
    y_2w = y_2w_f + y_2w_b
    desc0, desc1 = torch.chunk(y_2w, 2, dim=3)
    return desc0, desc1


def scan_vmamba(desc0, desc1):
    B, C, org_h, org_w = desc0.shape
    desc_2w = torch.cat([desc0, desc1], 3)
    desc_2h = torch.cat([desc0, desc1], 2)

    H = org_h
    W = desc_2w.shape[3]

    xs = desc_2w.new_empty((B, 4, C, H*W))

    xs[:, 0] = desc_2w.view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 1] = desc_2w.view(B, C, -1).flip([2])  # [w/2, 2w/2]
    xs[:, 2] = desc_2h.transpose(dim0=2, dim1=3).contiguous().view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 3] = desc_2h.transpose(dim0=2, dim1=3).contiguous().view(B, C, -1).flip([2])  # [w/2, 2w/2]

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_vmamba(ys, org_h, org_w):
    B, K, C, L = ys.shape

    y_2w_f = ys[:, 0].reshape(B, C, org_h, 2*org_w)
    y_2w_b = ys[:, 1].flip([2]).reshape(B, C, org_h, 2*org_w)
    y_2h_f = ys[:, 2].reshape(B, C, org_w, 2*org_h).transpose(2, 3)
    y_2h_b = ys[:, 3].flip([2]).reshape(B, C, org_w, 2*org_h).transpose(2, 3)
    y_2w, y_2h = y_2w_f + y_2w_b, y_2h_f + y_2h_b
    desc0_w, desc1_w = torch.chunk(y_2w, 2, dim=3)
    desc0_h, desc1_h = torch.chunk(y_2h, 2, dim=2)
    return desc0_w+desc0_h, desc1_w+desc1_h


def scan_evmamba(desc0, desc1, step_size):
    desc_2w, desc_2h = torch.cat([desc0, desc1], 3), torch.cat([desc0, desc1], 2)
    _, _, org_h, org_2w = desc_2w.shape
    B, C, org_2h, org_w = desc_2h.shape

    H = org_h // step_size
    W = org_2w // step_size

    xs = desc_2w.new_empty((B, 4, C, H*W))

    xs[:, 0] = desc_2w[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 1] = desc_2h.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)  # [w/2, 2w/2]
    xs[:, 2] = desc_2w[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 3] = desc_2h.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)  # [w/2, 2w/2]

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_evmamba(ys, ori_h: int, ori_w: int, step_size=2):
    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w = torch.zeros((B, C, new_h, 2*new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, new_h, 2*new_w))
    y_2h = torch.zeros((B, C, 2*new_h, new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, 2*new_h, new_w))

    y_2w[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, 1::step_size] = ys[:, 1].reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)
    y_2w[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, ::step_size] = ys[:, 3].reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()
    desc0_2w, desc1_2w = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, desc1_2h = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w+desc0_2h, desc1_2w+desc1_2h


class JointMamba(nn.Module):
    def __init__(self, feature_dim: int, depth,
                 ssm_cfg=None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 profiler=None):
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_layers = depth
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(create_block(
                    feature_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                ))
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.aggregator = GLU_3(feature_dim, feature_dim)

    def forward(self, data):
        desc0, desc1 = data['feat_8_0'], data['feat_8_1']
        desc0, desc1 = desc0.reshape(data['bs'], -1, data['h_8'], data['w_8']), desc1.reshape(data['bs'], -1, data['h_8'], data['w_8'])
        x, ori_h, ori_w = scan_jego(desc0, desc1, 2)
        for i in range(len(self.layers) // 4):
            y0 = self.layers[i * 4](x[:, 0])
            y1 = self.layers[i * 4 + 1](x[:, 1])
            y2 = self.layers[i * 4 + 2](x[:, 2])
            y3 = self.layers[i * 4 + 3](x[:, 3])
            y = torch.stack([y0, y1, y2, y3], 1).transpose(2, 3)
        desc0, desc1 = merge_jego(y, ori_h, ori_w, 2)
        desc = self.aggregator(torch.cat([desc0, desc1], 0))
        desc0, desc1 = torch.chunk(desc, 2, dim=0)
        desc0, desc1 = desc0.flatten(2, 3), desc1.flatten(2, 3)
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

# class MultiHeadMambaBlock(nn.Module):
#     """
#     Multi-Head Mamba Block for AS-Mamba architecture.
    
#     This block extends the standard Mamba block by splitting the output into
#     two heads: matching head and geometry head. This allows the model to
#     learn both appearance matching features and geometric transformation features
#     simultaneously.
    
#     Args:
#         dim (int): Input/output dimension for the matching head
#         d_geom (int, optional): Output dimension for the geometry head. 
#                                Defaults to dim // 2
#         mixer_cls: Mamba mixer class (typically partial(Mamba, ...))
#         norm_cls: Normalization class (LayerNorm or RMSNorm)
#         fused_add_norm (bool): Whether to use fused add+norm operation
#         residual_in_fp32 (bool): Whether to use fp32 for residual connection
#         drop_path (float): Drop path rate for regularization
#     """
    
#     def __init__(
#             self, 
#             dim: int, 
#             d_geom: int = None,
#             mixer_cls=None, 
#             norm_cls=nn.LayerNorm, 
#             fused_add_norm: bool = False, 
#             residual_in_fp32: bool = False,
#             drop_path: float = 0.,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.d_geom = d_geom if d_geom is not None else dim // 2
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
        
#         # Core Mamba mixer for sequence modeling
#         if mixer_cls is None:
#             mixer_cls = partial(Mamba, d_model=dim)
#         self.mixer = mixer_cls(dim)
        
#         # Normalization layer
#         self.norm = norm_cls(dim)
        
#         # Output projection heads
#         # Matching head: preserves full dimension for detailed matching features
#         self.out_proj_match = nn.Linear(dim, dim)
        
#         # Geometry head: projects to lower dimension for geometric features
#         self.out_proj_geom = nn.Linear(dim, self.d_geom)
        
#         # Optional: Layer normalization for each head output
#         self.norm_match = nn.LayerNorm(dim)
#         self.norm_geom = nn.LayerNorm(self.d_geom)
        
#         # Drop path for regularization (if specified)
#         self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        
#         # Ensure RMSNorm is available if fused_add_norm is requested
#         if self.fused_add_norm:
#             assert RMSNorm is not None, "RMSNorm import fails"
#             assert isinstance(
#                 self.norm, (nn.LayerNorm, RMSNorm)
#             ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

#     def forward(
#             self, 
#             x: torch.Tensor, 
#             inference_params=None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of the Multi-Head Mamba Block.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape (B, L, D)
#                             B = batch size, L = sequence length, D = dimension
#             inference_params: Optional parameters for inference mode
            
#         Returns:
#             y_match (torch.Tensor): Matching head output, shape (B, L, D)
#             y_geom (torch.Tensor): Geometry head output, shape (B, L, D_geom)
#         """
        
#         # Store residual connection for matching head
#         residual = x
        
#         # Apply normalization to input
#         hidden_states = self.norm(x.to(dtype=self.norm.weight.dtype))
        
#         # Convert residual to fp32 if specified (for numerical stability)
#         if self.residual_in_fp32:
#             residual = residual.to(torch.float32)
        
#         # Process through Mamba mixer
#         # The mixer learns long-range dependencies in the sequence
#         hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        
#         # Apply drop path regularization
#         hidden_states = self.drop_path(hidden_states)
        
#         # Split into two heads with different projections
#         # Matching head: focuses on appearance and texture features
#         y_match_pre = self.out_proj_match(hidden_states)
#         y_match = residual + y_match_pre  # Residual connection
#         y_match = self.norm_match(y_match)  # Normalize output
        
#         # Geometry head: focuses on spatial transformation and structure
#         # No residual connection as dimension changes
#         y_geom = self.out_proj_geom(hidden_states)
#         y_geom = self.norm_geom(y_geom)  # Normalize output
        
#         return y_match, y_geom

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         """Allocate cache for efficient inference."""
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class MultiHeadMambaBlock(nn.Module):
    """
    Multi-Head Mamba Block for initial feature splitting.
    
    USAGE: JointMambaMultiHead (Global Path)
    
    Architecture:
        Input (combined) → Mamba → Split into dual heads → Output (match, geom)
    
    This follows JamMa's original design where a single feature is processed
    and then split into matching and geometric representations.
    """
    
    def __init__(
        self, 
        dim: int, 
        d_geom: int = None,
        mixer_cls=None, 
        norm_cls=nn.LayerNorm, 
        fused_add_norm: bool = False, 
        residual_in_fp32: bool = False,
        drop_path: float = 0.,
    ):
        super().__init__()
        self.dim = dim
        self.d_geom = d_geom if d_geom is not None else dim // 2
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # Single Mamba mixer for combined processing
        if mixer_cls is None:
            mixer_cls = partial(Mamba, d_model=dim)
        self.mixer = mixer_cls(dim)
        
        # Normalization
        self.norm = norm_cls(dim)
        
        # Output projection heads (split into dual streams)
        self.out_proj_match = nn.Linear(dim, dim)
        self.out_proj_geom = nn.Linear(dim, self.d_geom)
        
        # Output normalization
        self.norm_match = nn.LayerNorm(dim)
        self.norm_geom = nn.LayerNorm(self.d_geom)
        
        # Regularization
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"

    def forward(
        self, 
        x: torch.Tensor, 
        inference_params=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: Single input → Dual output
        
        Args:
            x: Input features (B, L, D)
            inference_params: Optional inference parameters
            
        Returns:
            (y_match, y_geom): Dual-head outputs
        """
        residual = x
        hidden_states = self.norm(x.to(dtype=self.norm.weight.dtype))
        
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        
        # CRITICAL: Move to same device if needed (GPU mode fix)
        mixer_device = next(self.mixer.parameters()).device
        if hidden_states.device != mixer_device:
            hidden_states = hidden_states.to(mixer_device)
            residual = residual.to(mixer_device)
        
        # Process through Mamba
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = self.drop_path(hidden_states)
        
        # Split into dual heads
        y_match_pre = self.out_proj_match(hidden_states)
        y_match = residual + y_match_pre
        y_match = self.norm_match(y_match)
        
        y_geom = self.out_proj_geom(hidden_states)
        y_geom = self.norm_geom(y_geom)
        
        return y_match, y_geom

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_multihead_block(
    d_model,
    d_geom=None,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    block_type='single_input', 
):
    """
    Factory function to create a Multi-Head Mamba Block.
    
    This follows the same pattern as the original create_block function
    but returns our MultiHeadMambaBlock instead.
    """
    # if ssm_cfg is None:
    #     ssm_cfg = {}
    
    # factory_kwargs = {"device": device, "dtype": dtype}
    
    # # Create Mamba mixer with specified configuration
    # mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    
    # # Select normalization class
    # norm_cls = partial(
    #     nn.LayerNorm if not rms_norm else RMSNorm, 
    #     eps=norm_epsilon, 
    #     **factory_kwargs
    # )
    
    # # Create the multi-head block
    # block = MultiHeadMambaBlock(
    #     dim = d_model,
    #     d_geom=d_geom,
    #     mixer_cls=mixer_cls,
    #     fused_add_norm=fused_add_norm,
    #     residual_in_fp32=residual_in_fp32,
    #     drop_path=drop_path,
    #     rms_norm = rms_norm,
    #     norm_epsilon = norm_epsilon
    # )
    
    # # Store layer index for debugging/visualization
    # block.layer_idx = layer_idx
    
    # return block
    if ssm_cfg is None:
        ssm_cfg = {}
    
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, 
        eps=norm_epsilon, 
        **factory_kwargs
    )
    
    if block_type == 'dual_input':
        # For LocalAdaptiveMamba
        block = DualStreamMambaBlock(
            d_model=d_model,
            d_geom=d_geom or d_model // 2,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            drop_path=drop_path,
        )
    else:
        # For JointMambaMultiHead (default)
        block = MultiHeadMambaBlock(
            dim=d_model,
            d_geom=d_geom,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            drop_path=drop_path,
        )
    
    block.layer_idx = layer_idx
    return block

"""
Mamba modules for AS-Mamba architecture.

This file contains all Mamba-related components including:
- MultiHeadMambaBlock: Multi-head variant with matching and geometry outputs
- JointMambaMultiHead: Joint processing with JEGO strategy
- Helper functions for scanning and merging
"""

# ============= MultiHeadMambaBlock (from Step 2) =============

class MultiHeadMambaBlock(nn.Module):
    """
    Multi-Head Mamba Block for AS-Mamba architecture.
    
    This block extends the standard Mamba block by splitting the output into
    two heads: matching head and geometry head.
    """
    
    def __init__(
            self, 
            dim: int, 
            d_geom: int = None,
            mixer_cls=None, 
            norm_cls=nn.LayerNorm, 
            fused_add_norm: bool = False, 
            residual_in_fp32: bool = False,
            drop_path: float = 0.,
    ):
        super().__init__()
        self.dim = dim
        self.d_geom = d_geom if d_geom is not None else dim // 2
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # Core Mamba mixer
        if mixer_cls is None:
            mixer_cls = partial(Mamba, d_model=dim)
        self.mixer = mixer_cls(dim)
        
        # Normalization
        self.norm = norm_cls(dim)
        
        # Output projection heads
        self.out_proj_match = nn.Linear(dim, dim)
        self.out_proj_geom = nn.Linear(dim, self.d_geom)
        
        # Head-specific normalization
        self.norm_match = nn.LayerNorm(dim)
        self.norm_geom = nn.LayerNorm(self.d_geom)
        
        # Drop path
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, x: torch.Tensor, inference_params=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both matching and geometry features."""
        residual = x
        
        # Normalize and process
        hidden_states = self.norm(x.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        
        # Mamba processing
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = self.drop_path(hidden_states)
        
        # Split into two heads
        y_match_pre = self.out_proj_match(hidden_states)
        y_match = residual + y_match_pre
        y_match = self.norm_match(y_match)
        
        y_geom = self.out_proj_geom(hidden_states)
        y_geom = self.norm_geom(y_geom)
        
        return y_match, y_geom

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# def create_multihead_block(
#     d_model,
#     d_geom=None,
#     ssm_cfg=None,
#     norm_epsilon=1e-5,
#     drop_path=0.,
#     rms_norm=False,
#     residual_in_fp32=False,
#     fused_add_norm=False,
#     layer_idx=None,
#     device=None,
#     dtype=None,

# ):
#     """Factory function to create a Multi-Head Mamba Block."""
#     if ssm_cfg is None:
#         ssm_cfg = {}
    
#     factory_kwargs = {"device": device, "dtype": dtype}
#     mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
#     norm_cls = partial(
#         nn.LayerNorm if not rms_norm else RMSNorm, 
#         eps=norm_epsilon, 
#         **factory_kwargs
#     )
    
#     block = MultiHeadMambaBlock(
#         d_model,
#         d_geom=d_geom,
#         mixer_cls=mixer_cls,
#         norm_cls=norm_cls,
#         fused_add_norm=fused_add_norm,
#         residual_in_fp32=residual_in_fp32,
#         drop_path=drop_path,
#     )
#     block.layer_idx = layer_idx
#     return block


class DualStreamMambaBlock(nn.Module):
    """
    Dual-Stream Mamba Block for independent stream processing.
    
    USAGE: LocalAdaptiveMamba (Local Path with Adaptive Spans)
    
    Architecture:
        Input (match, geom) → Process independently → Output (match, geom)
    
    DESIGN RATIONALE:
    -----------------
    In LocalAdaptiveMamba, features are already separated into matching
    and geometric streams. Each stream needs independent processing with
    its own Mamba mixer to maintain stream-specific information.
    
    This is different from MultiHeadMambaBlock which CREATES the separation.
    Here we MAINTAIN the separation through parallel processing.
    
    WHY TWO SEPARATE MIXERS:
    ------------------------
    1. Different dimensionality: match (256), geom (64)
    2. Different semantic meaning: appearance vs. structure
    3. Independent learning dynamics for each modality
    4. Prevents information leakage between streams
    """
    
    def __init__(
        self,
        d_model: int,
        d_geom: int = 64,
        mixer_cls=None,
        norm_cls=nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_geom = d_geom
        self.residual_in_fp32 = residual_in_fp32
        
        # CRITICAL: TWO independent Mamba mixers
        # Matching stream: full dimension
        if mixer_cls is None:
            self.mixer_match = Mamba(d_model=d_model)
        else:
            # Use provided mixer_cls but override d_model
            self.mixer_match = Mamba(d_model=d_model)
        
        # Geometric stream: lower dimension
        self.mixer_geom = Mamba(d_model=d_geom)
        
        # Independent normalization for each stream
        self.norm_match = norm_cls(d_model)
        self.norm_geom = norm_cls(d_geom)
        
        # Regularization
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
    
    def forward(
        self,
        hidden_states_match: torch.Tensor,
        hidden_states_geom: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: Dual input → Dual output (independent processing)
        
        Args:
            hidden_states_match: Matching features (B, L, D_model)
            hidden_states_geom: Geometric features (B, L, D_geom) or None
            
        Returns:
            (y_match, y_geom): Independently processed outputs
        """
        
        # ========================================
        # MATCHING STREAM
        # ========================================
        residual_match = hidden_states_match
        
        # Normalize
        x_match = self.norm_match(
            hidden_states_match.to(dtype=self.norm_match.weight.dtype)
        )
        
        if self.residual_in_fp32:
            residual_match = residual_match.to(torch.float32)
        
        # CRITICAL: Device consistency (GPU mode)
        mixer_device = next(self.mixer_match.parameters()).device
        if x_match.device != mixer_device:
            x_match = x_match.to(mixer_device)
            residual_match = residual_match.to(mixer_device)
        
        # Process through Mamba (matching)
        y_match = self.mixer_match(x_match, inference_params=None)
        y_match = self.drop_path(y_match)
        y_match = residual_match + y_match
        
        # ========================================
        # GEOMETRIC STREAM
        # ========================================
        if hidden_states_geom is not None:
            residual_geom = hidden_states_geom
            
            # Normalize
            x_geom = self.norm_geom(
                hidden_states_geom.to(dtype=self.norm_geom.weight.dtype)
            )
            
            if self.residual_in_fp32:
                residual_geom = residual_geom.to(torch.float32)
            
            # Device consistency
            mixer_geom_device = next(self.mixer_geom.parameters()).device
            if x_geom.device != mixer_geom_device:
                x_geom = x_geom.to(mixer_geom_device)
                residual_geom = residual_geom.to(mixer_geom_device)
            
            # Process through Mamba (geometric)
            y_geom = self.mixer_geom(x_geom, inference_params=None)
            y_geom = self.drop_path(y_geom)
            y_geom = residual_geom + y_geom
        else:
            # Initialize if not provided
            B, L = hidden_states_match.shape[:2]
            y_geom = torch.zeros(
                B, L, self.d_geom,
                device=hidden_states_match.device,
                dtype=hidden_states_match.dtype
            )
        
        return y_match, y_geom
    
# ============= Standard Block (from original JamMa) =============

class Block(nn.Module):
    """Standard Mamba block for backward compatibility."""
    def __init__(
            self, 
            dim: int,
            d_geom: int = None,
            mixer_cls = None, 
            fused_add_norm: bool=False, 
            residual_in_fp32: bool =False, 
            drop_path: float =0.,
            rms_norm: bool = False,
            norm_epsilon: float = 1e-5
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        if rms_norm:
            self.norm = RMSNorm(dim, eps=norm_epsilon)
        else:
            self.norm = nn.LayerNorm(dim, eps=norm_epsilon)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, desc, inference_params=None):
        hidden_states = self.norm(desc.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            desc = desc.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return desc + hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model, ssm_cfg=None, norm_epsilon=1e-5, drop_path=0., rms_norm=False,
    residual_in_fp32=False, fused_add_norm=False, layer_idx=None,
    device=None, dtype=None, if_bimamba=False, bimamba_type="none",
    if_devide_out=False, init_layer_scale=None,
):
    """Factory function for standard block (backward compatibility)."""
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model, mixer_cls, norm_cls=norm_cls,
        fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# ============= JointMambaMultiHead (from Step 3) =============

class JointMambaMultiHead(nn.Module):
    """Joint Mamba module with Multi-Head support for AS-Mamba."""
    
    def __init__(
        self, 
        feature_dim: int, 
        depth: int,
        d_geom: int = None,
        return_geometry: bool = False,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        if_bimamba=False,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
        profiler=None
    ):
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_layers = depth
        self.feature_dim = feature_dim
        self.d_geom = d_geom if d_geom is not None else feature_dim // 4
        self.return_geometry = return_geometry
        
        # Create MultiHead Mamba blocks
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                create_multihead_block(
                    feature_dim,
                    d_geom=self.d_geom,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
            )
        
        # Initialize weights
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
        # Aggregators
        self.aggregator = GLU_3(feature_dim, feature_dim)
        if self.return_geometry:
            self.aggregator_geom = GLU_3(self.d_geom, self.d_geom)

    def forward(self, data):
        """Forward pass with JEGO scanning strategy."""
        desc0, desc1 = data['feat_8_0'], data['feat_8_1']
        desc0 = desc0.reshape(data['bs'], -1, data['h_8'], data['w_8'])
        desc1 = desc1.reshape(data['bs'], -1, data['h_8'], data['w_8'])
        
        # JEGO scanning
        x, ori_h, ori_w = scan_jego(desc0, desc1, 2)
        
        # Process through Multi-Head Mamba blocks
        y_match_list = []
        y_geom_list = []
        
        for i in range(len(self.layers) // 4):
            # Process 4 directional scans
            y0_match, y0_geom = self.layers[i * 4](x[:, 0])
            y1_match, y1_geom = self.layers[i * 4 + 1](x[:, 1])
            y2_match, y2_geom = self.layers[i * 4 + 2](x[:, 2])
            y3_match, y3_geom = self.layers[i * 4 + 3](x[:, 3])
            
            # Stack features
            y_match = torch.stack([y0_match, y1_match, y2_match, y3_match], 1)
            y_match_list.append(y_match.transpose(2, 3))
            
            if self.return_geometry:
                y_geom = torch.stack([y0_geom, y1_geom, y2_geom, y3_geom], 1)
                y_geom_list.append(y_geom.transpose(2, 3))
        
        # Merge and aggregate matching features
        # y_match = y_match_list[-1] if len(y_match_list) > 0 else torch.stack([y0_match, y1_match, y2_match, y3_match], 1).transpose(2, 3)
        y_match = y_match_list[-1] if y_match_list else x.transpose(2, 3)
        desc0_match, desc1_match = merge_jego(y_match, ori_h, ori_w, 2)
        desc_match = self.aggregator(torch.cat([desc0_match, desc1_match], 0))
        desc0_match, desc1_match = torch.chunk(desc_match, 2, dim=0)
        # desc0_match = desc0_match.flatten(2, 3)
        # desc1_match = desc1_match.flatten(2, 3)
        
        # Update data
        data.update({
            'feat_8_0': desc0_match.flatten(2, 3),
            'feat_8_1': desc1_match.flatten(2, 3),
        })
        
        # Optionally process geometric features
        # if self.return_geometry and len(y_geom_list) > 0:
        #     y_geom = y_geom_list[-1]
        #     desc0_geom, desc1_geom = merge_jego(y_geom, ori_h, ori_w, 2)
            
        #     if hasattr(self, 'aggregator_geom'):
        #         desc_geom = self.aggregator_geom(torch.cat([desc0_geom, desc1_geom], 0))
        #         desc0_geom, desc1_geom = torch.chunk(desc_geom, 2, dim=0)
            
        #     # desc0_geom = desc0_geom.flatten(2, 3)
        #     # desc1_geom = desc1_geom.flatten(2, 3)
            
        #     data.update({
        #         'feat_geom_0': desc0_geom,
        #         'feat_geom_1': desc1_geom,
        #     })
        if self.return_geometry:  # ★ 1. まず return_geometry が True かだけをチェック
            if len(y_geom_list) > 0:
            
                y_geom = y_geom_list[-1]
                desc0_geom, desc1_geom = merge_jego(y_geom, ori_h, ori_w, 2)
                
                if hasattr(self, 'aggregator_geom'):
                    desc_geom = self.aggregator_geom(torch.cat([desc0_geom, desc1_geom], 0))
                    desc0_geom, desc1_geom = torch.chunk(desc_geom, 2, dim=0)
            
            else:
                B, _, H, W = desc0.shape # 入力マッチング特徴量の形状を取得
                desc0_geom = torch.zeros(B, self.d_geom, H, W, device=desc0.device, dtype=desc0.dtype)
                desc1_geom = torch.zeros(B, self.d_geom, H, W, device=desc1.device, dtype=desc1.dtype)

            # このブロックは return_geometry=True なら必ず実行される
            data.update({
                'feat_geom_0': desc0_geom,
                'feat_geom_1': desc1_geom,
            })


# Backward compatible JointMamba (standard version)
class JointMamba(nn.Module):
    """Standard JointMamba for backward compatibility."""
    def __init__(self, feature_dim: int, depth, **kwargs):
        super().__init__()
        # ... (original JointMamba implementation from JamMa)
        # This would be the original implementation
        # For now, we can use JointMambaMultiHead with return_geometry=False
        self.impl = JointMambaMultiHead(feature_dim, depth, return_geometry=False, **kwargs)
    
    def forward(self, data):
        return self.impl(data)


# ============= Helper Functions =============

def _init_weights(
    module, n_layer, initializer_range=0.02,
    rescale_prenorm_residual=True, n_residuals_per_layer=1,
):
    """Initialize weights for Mamba blocks."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def scan_jego(desc0, desc1, step_size):
    """JEGO scanning strategy for joint processing."""
    desc_2w, desc_2h = torch.cat([desc0, desc1], 3), torch.cat([desc0, desc1], 2)
    _, _, org_h, org_2w = desc_2w.shape
    B, C, org_2h, org_w = desc_2h.shape

    H = org_h // step_size
    W = org_2w // step_size

    xs = desc_2w.new_empty((B, 4, C, H*W))

    xs[:, 0] = desc_2w[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
    xs[:, 1] = desc_2h.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)
    xs[:, 2] = desc_2w[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])
    xs[:, 3] = desc_2h.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_jego(ys, ori_h: int, ori_w: int, step_size=2):
    """JEGO merge strategy to reconstruct spatial structure."""
    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w = torch.zeros((B, C, new_h, 2*new_w), device=ys.device, dtype=ys.dtype)
    y_2h = torch.zeros((B, C, 2*new_h, new_w), device=ys.device, dtype=ys.dtype)

    y_2w[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, 1::step_size] = ys[:, 1].reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)
    y_2w[:, :, ::step_size, 1::step_size] = ys[:, 2].flip([2]).reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, ::step_size] = ys[:, 3].flip([2]).reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()
    
    desc0_2w, desc1_2w = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, desc1_2h = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w+desc0_2h, desc1_2w+desc1_2h
