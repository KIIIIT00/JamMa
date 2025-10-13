#!/usr/bin/env python3
"""
Master Test Runner for AS-Mamba

Runs all test suites and generates a comprehensive report.

Usage:
    python run_all_tests.py [OPTIONS]
    
Options:
    --quick         Run only fast tests
    --components    Run only component tests
    --integration   Run only integration tests
    --gpu          Force GPU tests (skip if no GPU)
    --save-report  Save test report to file
    
Examples:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --quick            # Quick smoke test
    python run_all_tests.py --save-report      # Save results
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title):
    """Print a section separator."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def check_dependencies():
    """Check if required dependencies are available."""
    print_section("Checking Dependencies")
    
    missing = []
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠ CUDA not available (CPU only)")
    except ImportError:
        missing.append("torch")
    
    # Check other dependencies
    deps = {
        'einops': 'einops',
        'numpy': 'numpy',
    }
    
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            missing.append(name)
            print(f"✗ {name} - MISSING")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n✓ All dependencies satisfied")
    return True


def run_component_tests():
    """Run component-level tests."""
    print_header("COMPONENT TESTS")
    
    try:
        from tests.test_components import run_all_tests
        success = run_all_tests()
        return success
    except Exception as e:
        print(f"❌ Failed to run component tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_integration_tests():
    """Run integration tests."""
    print_header("INTEGRATION TESTS")
    
    try:
        from tests.test_integration import run_all_tests
        success = run_all_tests()
        return success
    except Exception as e:
        print(f"❌ Failed to run integration tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_tests():
    """Run only quick smoke tests."""
    print_header("QUICK SMOKE TESTS")
    
    try:
        from tests.test_components import (
            test_flow_predictor,
            test_adaptive_span_computer,
        )
        
        results = {
            "FlowPredictor": test_flow_predictor(),
            "AdaptiveSpanComputer": test_adaptive_span_computer(),
        }
        
        passed = sum(results.values())
        total = len(results)
        
        print(f"\n{'='*80}")
        print(f"Quick Tests: {passed}/{total} passed")
        print(f"{'='*80}")
        
        return all(results.values())
        
    except Exception as e:
        print(f"❌ Failed to run quick tests: {e}")
        return False


def generate_report(results, duration, save_to_file=False):
    """Generate test report."""
    print_header("TEST REPORT")
    
    # Summary statistics
    total_tests = sum(len(tests) for tests in results.values())
    passed_tests = sum(
        sum(test_results.values()) 
        for test_results in results.values()
    )
    
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Build report
    report_lines = [
        "=" * 80,
        f"AS-Mamba Test Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration: {duration:.2f} seconds",
        "=" * 80,
        "",
        "Summary:",
        f"  Total Tests: {total_tests}",
        f"  Passed: {passed_tests}",
        f"  Failed: {total_tests - passed_tests}",
        f"  Pass Rate: {pass_rate:.1f}%",
        "",
    ]
    
    # Detailed results
    for suite_name, test_results in results.items():
        report_lines.append(f"\n{suite_name}:")
        report_lines.append("-" * 80)
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report_lines.append(f"  {test_name:.<60} {status}")
    
    report_lines.append("\n" + "=" * 80)
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file if requested
    if save_to_file:
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Report saved to: {report_file}")
    
    return pass_rate == 100.0


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="AS-Mamba Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run only quick smoke tests')
    parser.add_argument('--components', action='store_true',
                       help='Run only component tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU tests')
    parser.add_argument('--save-report', action='store_true',
                       help='Save test report to file')
    
    args = parser.parse_args()
    
    # Print banner
    print_header("AS-MAMBA TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # GPU check
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            print("\n❌ GPU tests requested but CUDA not available")
            sys.exit(1)
    
    # Run tests
    start_time = time.time()
    results = {}
    
    try:
        if args.quick:
            # Quick tests only
            success = run_quick_tests()
            results['Quick Tests'] = {'smoke_test': success}
            
        elif args.components:
            # Component tests only
            from tests.test_components import (
                test_flow_predictor,
                test_adaptive_span_computer,
                test_as_mamba_block,
                test_local_adaptive_mamba,
                test_feature_fusion_ffn,
            )
            
            results['Component Tests'] = {
                'FlowPredictor': test_flow_predictor(),
                'AdaptiveSpanComputer': test_adaptive_span_computer(),
                'AS_Mamba_Block': test_as_mamba_block(),
                'LocalAdaptiveMamba': test_local_adaptive_mamba(),
                'FeatureFusionFFN': test_feature_fusion_ffn(),
            }
            
        elif args.integration:
            # Integration tests only
            from tests.test_integration import (
                test_as_mamba_full_forward,
                test_loss_computation,
                test_gradient_flow,
                test_multiblock_flow_collection,
                test_memory_efficiency,
                test_batch_size_robustness,
                test_adaptive_span_diversity,
            )
            
            results['Integration Tests'] = {
                'Full Forward': test_as_mamba_full_forward(),
                'Loss Computation': test_loss_computation(),
                'Gradient Flow': test_gradient_flow(),
                'Multi-Block Flow': test_multiblock_flow_collection(),
                'Memory Efficiency': test_memory_efficiency(),
                'Batch Robustness': test_batch_size_robustness(),
                'Span Diversity': test_adaptive_span_diversity(),
            }
            
        else:
            # Run all tests
            component_success = run_component_tests()
            integration_success = run_integration_tests()
            
            results['All Tests'] = {
                'components': component_success,
                'integration': integration_success,
            }
    
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Generate report
    all_passed = generate_report(results, duration, args.save_report)
    
    # Exit with appropriate code
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Please review the report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()