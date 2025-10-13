"""
AS-Mamba Component-Level Unit Tests (Complete GPU-Compatible Version)

Tests individual components in isolation to verify correctness.
Supports both CPU and GPU modes.

Usage:
    python tests/test_component.py           # Auto-detect (GPU if available)
    python tests/test_component.py --cpu     # Force CPU mode
    
Or with pytest:
    pytest tests/test_component.py -v
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# CONFIGURATION
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
print("AS-MAMBA COMPONENT TESTS")
print("=" * 80)
print(f"Device: {'CPU' if FORCE_CPU else 'CUDA'}")
print(f"PyTorch: {torch.__version__}")


# ============================================================================
# HELPER FUNCTIONS
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
# TEST 1: Flow Predictor
# ============================================================================
def test_flow_predictor():
    """Test FlowPredictor with both MLP and KAN backends."""
    print("\n[TEST 1] FlowPredictor")
    print("-" * 40)
    
    try:
        from src.jamma.flow_predictor import FlowPredictor
        
        # Test parameters
        batch_size = 2
        d_model = 256
        d_geom = 64
        H, W = 32, 40
        
        device = get_device()
        print(f"Testing on: {device}")
        
        # Create dummy input
        feat_match = create_tensor(batch_size, d_model, H, W)
        feat_geom = create_tensor(batch_size, d_geom, H, W)
        
        # Test MLP version
        print("Testing MLP-based FlowPredictor...")
        predictor_mlp = FlowPredictor(
            d_model=d_model,
            d_geom=d_geom,
            hidden_dim=128,
            num_layers=3,
            use_kan=False
        ).to(device)
        
        with torch.no_grad():
            output_mlp = predictor_mlp(feat_match, feat_geom)
        
        assert 'flow' in output_mlp, "Missing 'flow' in output"
        assert 'uncertainty' in output_mlp, "Missing 'uncertainty' in output"
        assert 'flow_with_uncertainty' in output_mlp, "Missing 'flow_with_uncertainty' in output"
        
        flow = output_mlp['flow']
        uncertainty = output_mlp['uncertainty']
        flow_with_unc = output_mlp['flow_with_uncertainty']
        
        assert flow.shape == (batch_size, H, W, 2), f"Wrong flow shape: {flow.shape}"
        assert uncertainty.shape == (batch_size, H, W, 2), f"Wrong uncertainty shape: {uncertainty.shape}"
        assert flow_with_unc.shape == (batch_size, H, W, 4), f"Wrong combined shape: {flow_with_unc.shape}"
        
        print(f"✓ MLP FlowPredictor output shapes correct")
        print(f"  - Flow: {flow.shape}")
        print(f"  - Uncertainty: {uncertainty.shape}")
        print(f"  - Combined: {flow_with_unc.shape}")
        print(f"  - Device: {flow.device}")
        
        # Test KAN version (if available)
        print("\nTesting KAN-based FlowPredictor...")
        try:
            predictor_kan = FlowPredictor(
                d_model=d_model,
                d_geom=d_geom,
                hidden_dim=128,
                num_layers=3,
                use_kan=True
            ).to(device)
            
            with torch.no_grad():
                output_kan = predictor_kan(feat_match, feat_geom)
            assert output_kan['flow'].shape == (batch_size, H, W, 2)
            print(f"✓ KAN FlowPredictor works")
            
        except Exception as e:
            print(f"⚠ KAN FlowPredictor not available (optional): {e}")
        
        print("\n✅ TEST 1 PASSED: FlowPredictor")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 2: Adaptive Span Computer
# ============================================================================
def test_adaptive_span_computer():
    """Test AdaptiveSpanComputer."""
    print("\n[TEST 2] AdaptiveSpanComputer")
    print("-" * 40)
    
    try:
        from src.jamma.flow_predictor import AdaptiveSpanComputer
        
        batch_size = 2
        H, W = 32, 40
        device = get_device()
        
        print(f"Testing on: {device}")
        
        # Create dummy flow and uncertainty
        flow = create_tensor(batch_size, H, W, 2)
        uncertainty = create_tensor(batch_size, H, W, 2)
        
        # Create span computer
        span_computer = AdaptiveSpanComputer(
            base_span=7,
            max_span=15,
            temperature=1.0
        )
        
        # Compute spans
        with torch.no_grad():
            spans_x, spans_y = span_computer(flow, uncertainty)
        
        assert spans_x.shape == (batch_size, H, W), f"Wrong spans_x shape: {spans_x.shape}"
        assert spans_y.shape == (batch_size, H, W), f"Wrong spans_y shape: {spans_y.shape}"
        
        # Check value ranges
        assert spans_x.min() >= 3, "Span too small"
        assert spans_x.max() <= 15, "Span too large"
        
        print(f"✓ Span shapes correct: {spans_x.shape}")
        print(f"✓ Span range: [{spans_x.min().item():.1f}, {spans_x.max().item():.1f}]")
        print(f"✓ Mean span: {spans_x.float().mean().item():.2f}")
        print(f"✓ Device: {spans_x.device}")
        
        print("\n✅ TEST 2 PASSED: AdaptiveSpanComputer")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: AS-Mamba Block (Core Component)
# ============================================================================
def test_as_mamba_block():
    """Test AS_Mamba_Block with GPU support."""
    print("\n[TEST 3] AS_Mamba_Block")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba_block import AS_Mamba_Block
        
        # Test parameters
        batch_size = 2
        d_model = 256
        d_geom = 64
        H, W = 32, 40
        
        device = get_device()
        print(f"Testing on: {device}")
        
        # Create block
        print("Creating AS_Mamba_Block...")
        block = AS_Mamba_Block(
            d_model=d_model,
            d_geom=d_geom,
            d_ffn=512,
            global_depth=2,  # Reduced for faster testing
            local_depth=2,
            dropout=0.1,
            use_kan_flow=False
        ).to(device)
        
        block.eval()
        
        print(f"✓ Block created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in block.parameters()) / 1e6:.2f}M")
        print(f"  - Device: {next(block.parameters()).device}")
        
        # Prepare input data
        data = {
            'bs': batch_size,
            'h_8': H,
            'w_8': W,
            'feat_8_0': create_tensor(batch_size, d_model, H * W),
            'feat_8_1': create_tensor(batch_size, d_model, H * W),
        }
        
        print("\nRunning forward pass...")
        with torch.no_grad():
            output = block(data)
        
        # Check outputs
        assert 'feat_8_0' in output, "Missing feat_8_0 in output"
        assert 'feat_8_1' in output, "Missing feat_8_1 in output"
        assert 'feat_geom_0' in output, "Missing feat_geom_0 in output"
        assert 'feat_geom_1' in output, "Missing feat_geom_1 in output"
        assert 'flow_map' in output, "Missing flow_map in output"
        assert 'adaptive_spans' in output, "Missing adaptive_spans in output"
        
        # Check shapes
        assert output['feat_8_0'].shape == (batch_size, d_model, H * W)
        assert output['feat_geom_0'].shape == (batch_size, d_geom, H * W)
        assert output['flow_map'].shape == (2 * batch_size, H, W, 4)
        
        spans_x, spans_y = output['adaptive_spans']
        print(f"\n✓ Forward pass successful")
        print(f"  - Output feat shape: {output['feat_8_0'].shape}")
        print(f"  - Flow map shape: {output['flow_map'].shape}")
        print(f"  - Adaptive spans: {spans_x.shape}")
        print(f"  - Mean span: {spans_x.float().mean().item():.2f}")
        print(f"  - Output device: {output['feat_8_0'].device}")
        
        print("\n✅ TEST 3 PASSED: AS_Mamba_Block")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 4: LocalAdaptiveMamba
# ============================================================================
def test_local_adaptive_mamba():
    """Test LocalAdaptiveMamba with GPU support."""
    print("\n[TEST 4] LocalAdaptiveMamba")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba_block import LocalAdaptiveMamba
        
        batch_size = 2
        feature_dim = 256
        d_geom = 64
        H, W = 32, 40
        
        device = get_device()
        print(f"Testing on: {device}")
        
        print("Creating LocalAdaptiveMamba...")
        local_mamba = LocalAdaptiveMamba(
            feature_dim=feature_dim,
            depth=2,  # Reduced for testing
            d_geom=d_geom,
            dropout=0.1,
            max_span_groups=5
        ).to(device)
        
        local_mamba.eval()
        
        print(f"✓ LocalAdaptiveMamba created")
        print(f"  - Span groups: {local_mamba.span_groups}")
        print(f"  - Device: {next(local_mamba.parameters()).device}")
        
        # Create dummy inputs
        feat_match_0 = create_tensor(batch_size, feature_dim, H, W)
        feat_match_1 = create_tensor(batch_size, feature_dim, H, W)
        feat_geom_0 = create_tensor(batch_size, d_geom, H, W)
        feat_geom_1 = create_tensor(batch_size, d_geom, H, W)
        
        flow_map_0 = create_tensor(batch_size, H, W, 4)
        flow_map_1 = create_tensor(batch_size, H, W, 4)
        
        spans_x_0 = torch.randint(5, 12, (batch_size, H, W), device=device)
        spans_y_0 = torch.randint(5, 12, (batch_size, H, W), device=device)
        spans_x_1 = torch.randint(5, 12, (batch_size, H, W), device=device)
        spans_y_1 = torch.randint(5, 12, (batch_size, H, W), device=device)
        
        print("\nRunning forward pass...")
        with torch.no_grad():
            out_match_0, out_match_1, out_geom_0, out_geom_1 = local_mamba(
                feat_match_0, feat_match_1,
                feat_geom_0, feat_geom_1,
                flow_map_0, flow_map_1,
                spans_x_0, spans_y_0,
                spans_x_1, spans_y_1
            )
        
        # Check outputs
        assert out_match_0.shape == (batch_size, feature_dim, H, W), \
            f"Wrong shape: {out_match_0.shape}"
        assert out_geom_0.shape == (batch_size, d_geom, H, W), \
            f"Wrong shape: {out_geom_0.shape}"
        
        print(f"✓ Forward pass successful")
        print(f"  - Output match shape: {out_match_0.shape}")
        print(f"  - Output geom shape: {out_geom_0.shape}")
        print(f"  - Output device: {out_match_0.device}")
        
        print("\n✅ TEST 4 PASSED: LocalAdaptiveMamba")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: Feature Fusion FFN
# ============================================================================
def test_feature_fusion_ffn():
    """Test FeatureFusionFFN with GPU support."""
    print("\n[TEST 5] FeatureFusionFFN")
    print("-" * 40)
    
    try:
        from src.jamma.as_mamba_block import FeatureFusionFFN
        
        batch_size = 2
        d_model = 256
        H, W = 32, 40
        n_sources = 3
        
        device = get_device()
        print(f"Testing on: {device}")
        
        # Create FFN
        ffn = FeatureFusionFFN(d_model=d_model, d_ffn=512, dropout=0.1).to(device)
        
        # Create input (stacked features from multiple sources)
        combined_features = create_tensor(batch_size, n_sources, d_model, H, W)
        
        # Forward pass
        with torch.no_grad():
            output = ffn(combined_features)
        
        assert output.shape == (batch_size, d_model, H, W)
        
        print(f"✓ Input shape: {combined_features.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Device: {output.device}")
        
        print("\n✅ TEST 5 PASSED: FeatureFusionFFN")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Test Runner
# ============================================================================
def run_all_tests():
    """Run all component tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL COMPONENT TESTS")
    print("=" * 80)
    
    results = {
        "FlowPredictor": test_flow_predictor(),
        "AdaptiveSpanComputer": test_adaptive_span_computer(),
        "AS_Mamba_Block": test_as_mamba_block(),
        "LocalAdaptiveMamba": test_local_adaptive_mamba(),
        "FeatureFusionFFN": test_feature_fusion_ffn(),
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
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