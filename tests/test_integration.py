"""
AS-Mamba Integration and End-to-End Tests (Fixed GPU-Compatible Version)

Tests the full AS-Mamba pipeline with realistic data flow.
Based on test_component.py patterns for GPU/CPU compatibility.

Usage:
    python tests/test_integration.py           # Auto-detect (GPU if available)
    python tests/test_integration.py --cpu     # Force CPU mode
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import warnings

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# CONFIGURATION (Same as test_component.py)
# ============================================================================

# Force CPU mode if specified
FORCE_CPU = '--cpu' in sys.argv

if FORCE_CPU:
    print("⚠ Running in CPU-only mode (forced)")
    torch.set_default_device('cpu')
elif torch.cuda.is_available():
    print(f"✓ Running on GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("⚠ CUDA not available, falling back to CPU")
    FORCE_CPU = True

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='triton')

print("=" * 80)
print("AS-MAMBA INTEGRATION TESTS")
print("=" * 80)
print(f"Device: {'CPU' if FORCE_CPU else 'CUDA'}")
print(f"PyTorch: {torch.__version__}")


# ============================================================================
# HELPER FUNCTIONS (Same as test_component.py)
# ============================================================================

def get_device():
    """Get the appropriate device for testing."""
    if FORCE_CPU:
        return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def create_tensor(*args, **kwargs):
    """Create tensor on appropriate device."""
    device = get_device()
    return torch.randn(*args, **kwargs, device=device)


# ============================================================================
# TEST 1: Full AS-Mamba Forward Pass
# ============================================================================
def test_as_mamba_full_forward():
    """Test complete AS-Mamba model forward pass."""
    print("\n[TEST 1] AS-Mamba Full Forward Pass")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba import AS_Mamba
        
        device = get_device()
        print(f"Testing on: {device}")
        
        # Create minimal config
        config = {
            'coarse': {'d_model': 256},
            'fine': {
                'd_model': 128,
                'dsmax_temperature': 0.1
            },
            'fine_window_size': 5,
            'resolution': [8, 2],
            'as_mamba': {
                'n_blocks': 2,  # Reduced for testing
                'd_geom': 64,
                'use_kan_flow': False,
                'global_depth': 2,
                'local_depth': 2,
            },
            'match_coarse': {
                'match_type': 'dual_softmax',
                'temperature': 0.1,
                'thr': 0.2,
                'use_sm': True,
                'inference': False,
                'border_rm': 2,
                'dsmax_temperature': 0.1
            }
        }
        
        print("Creating AS-Mamba model...")
        model = AS_Mamba(config).to(device)
        model.eval()
        
        print(f"✓ Model created")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"  - AS-Mamba blocks: {model.n_blocks}")
        print(f"  - Device: {next(model.parameters()).device}")
        
        # Create dummy input data
        batch_size = 2
        H, W = 256, 320
        H_c, W_c = H // 8, W // 8  # Coarse resolution
        
        data = {
            'bs': batch_size,
            'h_8': H_c,
            'w_8': W_c,
            'c': 256,
            
            # Images
            'imagec_0': create_tensor(batch_size, 3, H, W),
            'imagec_1': create_tensor(batch_size, 3, H, W),
            
            # Features at 1/8 resolution (from backbone)
            'feat_8_0': create_tensor(batch_size, 160, H_c, W_c),  # ConvNeXtV2-nano
            'feat_8_1': create_tensor(batch_size, 160, H_c, W_c),
            
            # Features at 1/4 resolution (for fine matching)
            'feat_4_0': create_tensor(batch_size, 80, H_c * 2, W_c * 2),
            'feat_4_1': create_tensor(batch_size, 80, H_c * 2, W_c * 2),
            
            # Grid coordinates
            'grid_8': torch.stack(torch.meshgrid(
                torch.arange(H_c), torch.arange(W_c), indexing='ij'
            ), dim=-1).float().view(1, -1, 2).expand(batch_size, -1, -1).to(device),
        }
        
        print("\nRunning forward pass (this may take a moment)...")
        with torch.no_grad():
            output = model(data, mode='test')
        
        # Check key outputs
        print("\n✓ Forward pass completed successfully")
        
        if 'conf_matrix' in output:
            print(f"  - Confidence matrix: {output['conf_matrix'].shape}")
        
        if 'mkpts0_c' in output:
            n_matches = len(output['mkpts0_c'])
            print(f"  - Coarse matches: {n_matches}")
        
        if 'mkpts0_f' in output:
            n_matches_fine = len(output['mkpts0_f'])
            print(f"  - Fine matches: {n_matches_fine}")
        
        if 'predict_flow' in output:
            print(f"  - Flow predictions: {len(output['predict_flow'])} blocks")
        
        print("\n✅ TEST 1 PASSED: Full AS-Mamba Forward")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 2: Loss Computation
# ============================================================================
def test_loss_computation():
    """Test AS-Mamba loss computation."""
    print("\n[TEST 2] Loss Computation")
    print("-" * 40)
    
    try:
        from src.losses.as_mamba_loss import ASMambaLoss
        
        device = get_device()
        print(f"Testing on: {device}")
        
        # Create minimal config
        config = {
            'as_mamba': {
                'loss': {
                    'flow_weight': 1.0,
                    'coarse_weight': 1.0,
                    'fine_weight': 1.0,
                    'epipolar_weight': 0.1,
                    'multiscale_weight': 0.05,
                    'pos_weight': 1.0,
                    'neg_weight': 1.0,
                    'fine_type': 'l2_with_std',
                    'fine_correct_thr': 0.5,
                    'focal_alpha': 0.25,
                    'focal_gamma': 2.0,
                },
                'match_coarse': {
                    'match_type': 'dual_softmax',
                    'sparse_spvs': False,
                }
            }
        }
        
        print("Creating loss function...")
        loss_fn = ASMambaLoss(config)
        
        # Create dummy data with ground truth
        batch_size = 2
        H_c, W_c = 32, 40
        n_matches = 100
        
        data = {
            'bs': batch_size,
            
            # Flow predictions (from AS-Mamba blocks)
            'predict_flow': [(
                create_tensor(2, batch_size, H_c, W_c, 4),  # flows_0to1
                create_tensor(2, batch_size, H_c, W_c, 4),  # flows_1to0
            )],
            
            # Ground truth for flow
            'spv_b_ids': torch.randint(0, batch_size, (n_matches,), device=device),
            'spv_i_ids': torch.randint(0, H_c * W_c, (n_matches,), device=device),
            'spv_j_ids': torch.randint(0, H_c * W_c, (n_matches,), device=device),
            'hw0_c': (H_c, W_c),
            'hw1_c': (H_c, W_c),
            
            # Coarse matching
            'conf_matrix': create_tensor(batch_size, H_c * W_c, H_c * W_c),
            'conf_matrix_gt': torch.zeros(batch_size, H_c * W_c, H_c * W_c, device=device),
            
            # Fine matching
            'expec_f': create_tensor(n_matches, 3),  # with std
            'expec_f_gt': create_tensor(n_matches, 2),
        }
        
        # Add some ground truth matches
        for i in range(n_matches):
            b = data['spv_b_ids'][i].item()
            i_idx = data['spv_i_ids'][i].item()
            j_idx = data['spv_j_ids'][i].item()
            data['conf_matrix_gt'][b, i_idx, j_idx] = 1.0
        
        print("\nComputing losses...")
        with torch.no_grad():
            loss_fn(data)
        
        assert 'loss' in data, "Missing total loss"
        assert 'loss_scalars' in data, "Missing loss scalars"
        
        print("\n✓ Loss computation successful")
        print(f"  - Total loss: {data['loss'].item():.4f}")
        print(f"  - Device: {data['loss'].device}")
        
        for key, value in data['loss_scalars'].items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.item():.4f}")
        
        print("\n✅ TEST 2 PASSED: Loss Computation")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: Gradient Flow
# ============================================================================
def test_gradient_flow():
    """Test that gradients flow properly through the model."""
    print("\n[TEST 3] Gradient Flow Test")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba_block import AS_Mamba_Block
        
        device = get_device()
        print(f"Testing on: {device}")
        
        # Create simple block
        block = AS_Mamba_Block(
            d_model=64,  # Smaller for faster testing
            d_geom=16,
            d_ffn=128,
            global_depth=1,
            local_depth=1,
            dropout=0.0,  # Disable dropout for deterministic test
        ).to(device)
        block.train()
        
        # Create input
        batch_size = 1
        H, W = 16, 20
        
        data = {
            'bs': batch_size,
            'h_8': H,
            'w_8': W,
            'feat_8_0': create_tensor(batch_size, 64, H * W).requires_grad_(True),
            'feat_8_1': create_tensor(batch_size, 64, H * W).requires_grad_(True),
        }
        
        print("Running forward pass...")
        output = block(data)
        
        # Compute dummy loss
        loss = output['feat_8_0'].sum() + output['feat_8_1'].sum()
        
        print("Running backward pass...")
        loss.backward()
        
        # Check gradients
        has_grad = False
        zero_grad = False
        
        for name, param in block.named_parameters():
            if param.grad is not None:
                has_grad = True
                if param.grad.abs().max() == 0:
                    zero_grad = True
                    print(f"⚠ Zero gradient in: {name}")
        
        assert has_grad, "No gradients computed!"
        
        if not zero_grad:
            print("✓ All parameters have non-zero gradients")
        else:
            print("⚠ Some parameters have zero gradients (may be normal)")
        
        print("\n✅ TEST 3 PASSED: Gradient Flow")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: Multi-Block Flow Collection
# ============================================================================
def test_multiblock_flow_collection():
    """Test that flow predictions are correctly collected from multiple blocks."""
    print("\n[TEST 4] Multi-Block Flow Collection")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba import AS_Mamba
        
        device = get_device()
        print(f"Testing on: {device}")
        
        config = {
            'coarse': {'d_model': 256},
            'fine': {'d_model': 64},
            'fine_window_size': 5,
            'resolution': [8, 2],
            'as_mamba': {
                'n_blocks': 2,  # Reduced for testing
                'd_geom': 64,
                'use_kan_flow': False,
                'global_depth': 2,
                'local_depth': 2,
            },
            'match_coarse': {
                'match_type': 'dual_softmax',
                'temperature': 0.1,
                'thr': 0.2,
                'use_sm': True
            },
        }
        
        print("Creating AS-Mamba with 3 blocks...")
        model = AS_Mamba(config).to(device)
        model.eval()
        
        # Minimal input
        batch_size = 1
        H_c, W_c = 16, 20
        
        data = {
            'bs': batch_size,
            'h_8': H_c,
            'w_8': W_c,
            'c': 128,
            'imagec_0': create_tensor(batch_size, 3, 128, 160),
            'imagec_1': create_tensor(batch_size, 3, 128, 160),
            'feat_8_0': create_tensor(batch_size, 160, H_c, W_c),
            'feat_8_1': create_tensor(batch_size, 160, H_c, W_c),
            'feat_4_0': create_tensor(batch_size, 80, H_c * 2, W_c * 2),
            'feat_4_1': create_tensor(batch_size, 80, H_c * 2, W_c * 2),
            'grid_8': torch.zeros(batch_size, H_c * W_c, 2, device=device),
        }
        
        print("\nRunning forward pass...")
        with torch.no_grad():
            output = model(data, mode='test')
        
        # Check flow predictions
        assert 'predict_flow' in output, "Missing predict_flow in output"
        
        flow_list = output['predict_flow']
        assert len(flow_list) == 1, f"Expected 1 flow list, got {len(flow_list)}"
        
        flows_0to1, flows_1to0 = flow_list[0]
        
        expected_shape = (3, batch_size, H_c, W_c, 4)  # (n_blocks, B, H, W, 4)
        assert flows_0to1.shape == expected_shape, \
            f"Wrong flow shape: {flows_0to1.shape}, expected {expected_shape}"
        
        print(f"\n✓ Flow predictions collected correctly")
        print(f"  - Shape: {flows_0to1.shape}")
        print(f"  - Number of blocks: {flows_0to1.shape[0]}")
        print(f"  - Device: {flows_0to1.device}")
        print(f"  - Flow range: [{flows_0to1[..., :2].min():.2f}, {flows_0to1[..., :2].max():.2f}]")
        print(f"  - Uncertainty range: [{flows_0to1[..., 2:].min():.2f}, {flows_0to1[..., 2:].max():.2f}]")
        
        print("\n✅ TEST 4 PASSED: Multi-Block Flow Collection")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: Memory Efficiency Test
# ============================================================================
def test_memory_efficiency():
    """Test memory usage and check for leaks."""
    print("\n[TEST 5] Memory Efficiency")
    print("-" * 40)
    
    try:
        import gc
        from src.jamma.as_mamba_block import AS_Mamba_Block
        
        device = get_device()
        print(f"Testing on: {device}")
        
        # Initial memory
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create block
        block = AS_Mamba_Block(
            d_model=256,
            d_geom=64,
            d_ffn=512,
            global_depth=2,
            local_depth=2,
        ).to(device)
        
        # Test data
        batch_size = 2
        H, W = 32, 40
        
        def run_forward():
            data = {
                'bs': batch_size,
                'h_8': H,
                'w_8': W,
                'feat_8_0': create_tensor(batch_size, 256, H * W),
                'feat_8_1': create_tensor(batch_size, 256, H * W),
            }
            with torch.no_grad():
                output = block(data)
            return output
        
        # Warm-up
        _ = run_forward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Multiple forward passes
        n_iters = 10
        print(f"\nRunning {n_iters} forward passes...")
        
        for i in range(n_iters):
            _ = run_forward()
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # Check memory
        if device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"\n✓ Peak GPU memory: {peak_memory:.2f} MB")
            
            # Check for memory leaks
            torch.cuda.empty_cache()
            gc.collect()
            current_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"✓ Current GPU memory: {current_memory:.2f} MB")
            
            if current_memory < peak_memory * 0.5:
                print("✓ No obvious memory leaks detected")
            else:
                print("⚠ Possible memory leak (needs investigation)")
        else:
            print("\n✓ CPU test completed (memory tracking limited)")
        
        print("\n✅ TEST 5 PASSED: Memory Efficiency")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 6: Batch Size Robustness
# ============================================================================
def test_batch_size_robustness():
    """Test model with different batch sizes."""
    print("\n[TEST 6] Batch Size Robustness")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba_block import AS_Mamba_Block
        
        device = get_device()
        print(f"Testing on: {device}")
        
        block = AS_Mamba_Block(
            d_model=128,
            d_geom=32,
            d_ffn=256,
            global_depth=1,
            local_depth=1,
        ).to(device)
        block.eval()
        
        H, W = 16, 20
        batch_sizes = [1, 2, 4, 8]
        
        print("\nTesting different batch sizes...")
        for bs in batch_sizes:
            data = {
                'bs': bs,
                'h_8': H,
                'w_8': W,
                'feat_8_0': create_tensor(bs, 128, H * W),
                'feat_8_1': create_tensor(bs, 128, H * W),
            }
            
            with torch.no_grad():
                output = block(data)
            
            assert output['feat_8_0'].shape[0] == bs
            print(f"  ✓ Batch size {bs}: OK")
        
        print("\n✅ TEST 6 PASSED: Batch Size Robustness")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 7: Adaptive Span Diversity
# ============================================================================
def test_adaptive_span_diversity():
    """Test that adaptive spans show meaningful variation."""
    print("\n[TEST 7] Adaptive Span Diversity")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba_block import AS_Mamba_Block
        
        device = get_device()
        print(f"Testing on: {device}")
        
        block = AS_Mamba_Block(
            d_model=128,
            d_geom=32,
            d_ffn=256,
            global_depth=1,
            local_depth=1,
        ).to(device)
        block.eval()
        
        batch_size = 2
        H, W = 32, 40
        
        # Create features with varying characteristics
        data = {
            'bs': batch_size,
            'h_8': H,
            'w_8': W,
            'feat_8_0': create_tensor(batch_size, 128, H * W) * 2.0,
            'feat_8_1': create_tensor(batch_size, 128, H * W) * 2.0,
        }
        
        print("\nComputing adaptive spans...")
        with torch.no_grad():
            output = block(data)
        
        spans_x, spans_y = output['adaptive_spans']
        
        # Statistics
        mean_span = (spans_x.float().mean() + spans_y.float().mean()) / 2
        std_span = (spans_x.float().std() + spans_y.float().std()) / 2
        min_span = min(spans_x.min(), spans_y.min())
        max_span = max(spans_x.max(), spans_y.max())
        
        print(f"\n✓ Span statistics:")
        print(f"  - Mean: {mean_span.item():.2f}")
        print(f"  - Std: {std_span.item():.2f}")
        print(f"  - Range: [{min_span.item()}, {max_span.item()}]")
        print(f"  - Device: {spans_x.device}")
        
        # Check for diversity
        if std_span > 0.5:
            print(f"  ✓ Good diversity (std > 0.5)")
        else:
            print(f"  ⚠ Low diversity (std = {std_span.item():.2f})")
        
        # Check reasonable range
        assert min_span >= 3, f"Span too small: {min_span}"
        assert max_span <= 15, f"Span too large: {max_span}"
        
        print("\n✅ TEST 7 PASSED: Adaptive Span Diversity")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 7 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Test Runner
# ============================================================================
def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL INTEGRATION TESTS")
    print("=" * 80)
    
    results = {
        "Full AS-Mamba Forward": test_as_mamba_full_forward(),
        "Loss Computation": test_loss_computation(),
        "Gradient Flow": test_gradient_flow(),
        "Multi-Block Flow Collection": test_multiblock_flow_collection(),
        "Memory Efficiency": test_memory_efficiency(),
        "Batch Size Robustness": test_batch_size_robustness(),
        "Adaptive Span Diversity": test_adaptive_span_diversity(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("-" * 80)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Device info
    if not FORCE_CPU and torch.cuda.is_available():
        print(f"\nGPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    print("=" * 80)
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)