"""
Integration test for AS-Mamba model.
Tests model instantiation and forward pass with dummy data.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.as_mamba_default import get_cfg_defaults
from src.jamma.as_mamba import AS_Mamba
from src.jamma.backbone import CovNextV2_nano


def create_dummy_data(batch_size=2, height=256, width=256, device='cuda'):
    """Create dummy input data for testing."""
    data = {
        # Image data
        'imagec_0': torch.randn(batch_size, 3, height, width).to(device),
        'imagec_1': torch.randn(batch_size, 3, height, width).to(device),
        
        # Metadata
        'bs': batch_size,
        'dataset_name': ['test'],
        'pair_names': [('img0', 'img1')] * batch_size,
    }
    return data


def test_backbone():
    """Test CNN backbone."""
    print("Testing CNN Backbone...")
    backbone = CovNextV2_nano()
    
    # Create dummy data
    data = create_dummy_data(batch_size=1, device='cpu')
    
    # Forward pass
    try:
        backbone(data)
        print("✓ Backbone forward pass successful")
        print(f"  - feat_8_0 shape: {data['feat_8_0'].shape}")
        print(f"  - feat_8_1 shape: {data['feat_8_1'].shape}")
        print(f"  - feat_4_0 shape: {data['feat_4_0'].shape}")
        print(f"  - feat_4_1 shape: {data['feat_4_1'].shape}")
    except Exception as e:
        print(f"✗ Backbone test failed: {e}")
        return False
    return True


def test_as_mamba_model():
    """Test complete AS-Mamba model."""
    print("\nTesting AS-Mamba Model...")
    
    # Load config
    config = get_cfg_defaults()
    
    # Convert to dict format expected by model
    config_dict = {
        'coarse': {'d_model': config.AS_MAMBA.COARSE.D_MODEL},
        'fine': {'d_model': config.AS_MAMBA.FINE.D_MODEL,
                 'dsmax_temperature': 0.1,
                 'inference': True,
                 'thr': 0.2,
                 },
        'resolution': config.AS_MAMBA.RESOLUTION,
        'fine_window_size': config.AS_MAMBA.FINE_WINDOW_SIZE,
        'match_coarse': {
            'thr': 0.2, 'use_sm': True, 'border_rm': 2,
            'dsmax_temperature': 0.1, 'inference': True
        },
        'as_mamba': {
            'n_blocks': 1,  # Reduce for memory test
            'd_geom': 32,
            'use_kan_flow': False,
            'global_depth': 1,
            'local_depth': 1,
            'use_geom_for_fine': False,
            'window_size': 2 
        }
    }
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = AS_Mamba(config_dict)
        model = model.to(device)
        model.eval()
        print("✓ Model instantiation successful")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {n_params/1e6:.2f}M")
        
        # Count AS-Mamba blocks
        print(f"  - Number of AS-Mamba blocks: {len(model.as_mamba_blocks)}")
        
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Create dummy data with backbone
    backbone = CovNextV2_nano().to(device)
    data = create_dummy_data(batch_size=2, device=device)
    
    try:
        with torch.no_grad():
            # Extract features with backbone
            backbone(data)
            
            # Forward through AS-Mamba
            model(data, mode='test')
            
        print("✓ Forward pass successful")
        
        # Check outputs
        if 'mkpts0_c' in data:
            print(f"  - Coarse matches: {data['mkpts0_c'].shape[0]}")
        if 'mkpts0_f' in data:
            print(f"  - Fine matches: {data['mkpts0_f'].shape[0]}")
        if 'as_mamba_flow' in data:
            print(f"  - Flow map shape: {data['as_mamba_flow'].shape}")
        if 'feat_geom_0' in data:
            print(f"  - Geometry features shape: {data['feat_geom_0'].shape}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_memory_usage():
    """Test memory usage with different batch sizes."""
    print("\nTesting memory usage...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    config = get_cfg_defaults()
    config_dict = {
        'coarse': {'d_model': config.AS_MAMBA.COARSE.D_MODEL},
        'fine': {'d_model': config.AS_MAMBA.FINE.D_MODEL,
                 'dsmax_temperature': 0.1,
                 'inference': True,
                 'thr': 0.2,
                 },
        'resolution': config.AS_MAMBA.RESOLUTION,
        'fine_window_size': config.AS_MAMBA.FINE_WINDOW_SIZE,
        'match_coarse': {
            'thr': 0.2, 'use_sm': True, 'border_rm': 2,
            'dsmax_temperature': 0.1, 'inference': True
        },
        'as_mamba': {
            'n_blocks': 1,  # Reduce for memory test
            'd_geom': 32,
            'use_kan_flow': False,
            'global_depth': 1,
            'local_depth': 1,
            'use_geom_for_fine': False,
            'window_size': 2
        }
    }
    
    model = AS_Mamba(config_dict).cuda().eval()
    backbone = CovNextV2_nano().cuda().eval()
    
    batch_sizes = [1, 2, 4]
    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            data = create_dummy_data(batch_size=bs, device='cuda')
            
            with torch.no_grad():
                backbone(data)
                model(data, mode='test')
            
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  - Batch size {bs}: {memory_mb:.1f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  - Batch size {bs}: OOM")
                break
            else:
                raise e


def main():
    """Run all tests."""
    print("="*50)
    print("AS-Mamba Integration Tests")
    print("="*50)
    
    tests = [
        ("Backbone", test_backbone),
        ("AS-Mamba Model", test_as_mamba_model),
        ("Memory Usage", test_memory_usage)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*30}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(success for _, success in results if success is not False)
    if all_passed:
        print("\n🎉 All tests passed! AS-Mamba is ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)