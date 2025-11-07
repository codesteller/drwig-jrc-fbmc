#!/usr/bin/env python3
"""
FBMC-TTDF Validation Test Script

Quick verification that the framework is working correctly.
Run this after installation to ensure everything is set up properly.
"""

import sys
import os
import time
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported."""
    print("🔍 Checking imports...")
    
    try:
        import numpy as np
        print(f"   ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"   ✗ numpy failed: {e}")
        return False
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        import matplotlib.pyplot as plt
        print(f"   ✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"   ✗ matplotlib failed: {e}")
        return False
    
    try:
        import scipy
        print(f"   ✓ scipy {scipy.__version__}")
    except ImportError as e:
        print(f"   ✗ scipy failed: {e}")
        return False
    
    try:
        import fbmc_ofdm_validation as fv
        print(f"   ✓ fbmc_ofdm_validation module")
    except ImportError as e:
        print(f"   ✗ fbmc_ofdm_validation failed: {e}")
        return False
    
    return True

def check_environment():
    """Check Python version and environment."""
    print("🐍 Checking Python environment...")
    
    # Check Python version
    version = sys.version_info
    if version.major >= 3 and version.minor >= 12:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"   ⚠️  Python {version.major}.{version.minor}.{version.micro} (3.12+ recommended)")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✓ Virtual environment detected")
    else:
        print("   ⚠️  No virtual environment detected (recommended)")
    
    return True

def test_basic_functionality():
    """Test basic framework functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Import the main module
        import fbmc_ofdm_validation as fv
        
        # Test OFDM class instantiation
        ofdm = fv.OFDM(N_subcarriers=64)  # Small size for quick test
        print("   ✓ OFDM class instantiation")
        
        # Test FBMC class instantiation
        fbmc = fv.FBMC(N_subcarriers=64, K_overlapping=4)
        print("   ✓ FBMC class instantiation")
        
        # Test symbol generation
        symbols = ofdm.generate_qam_symbols(M=16, num_symbols=5)
        assert symbols.shape == (64, 5), f"Wrong symbol shape: {symbols.shape}"
        print("   ✓ QAM symbol generation")
        
        # Test OFDM modulation
        ofdm_signal = ofdm.modulate(symbols)
        expected_len = 5 * (64 + int(64 * 0.25))  # 5 symbols with CP
        assert len(ofdm_signal) == expected_len, f"Wrong OFDM signal length: {len(ofdm_signal)}"
        print("   ✓ OFDM modulation")
        
        # Test FBMC modulation
        fbmc_signal = fbmc.modulate(symbols)
        assert len(fbmc_signal) > 0, "FBMC signal is empty"
        print("   ✓ FBMC modulation")
        
        # Test spectrum computation
        ofdm_spectrum = ofdm.compute_spectrum(ofdm_signal, nfft=512)
        assert len(ofdm_spectrum) == 512, f"Wrong spectrum length: {len(ofdm_spectrum)}"
        print("   ✓ Spectrum computation")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Basic functionality test failed: {e}")
        return False

def test_directory_structure():
    """Check if required directories exist and are writable."""
    print("📁 Checking directory structure...")
    
    # Check if logs directory exists or can be created
    logs_dir = Path("logs")
    if not logs_dir.exists():
        try:
            logs_dir.mkdir(exist_ok=True)
            print("   ✓ Created logs directory")
        except Exception as e:
            print(f"   ✗ Cannot create logs directory: {e}")
            return False
    else:
        print("   ✓ Logs directory exists")
    
    # Test write permissions
    test_file = logs_dir / "test_write.tmp"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print("   ✓ Write permissions OK")
    except Exception as e:
        print(f"   ✗ Cannot write to logs directory: {e}")
        return False
    
    return True

def run_quick_validation():
    """Run a quick version of the main validation."""
    print("🚀 Running quick validation test...")
    
    try:
        import fbmc_ofdm_validation as fv
        import matplotlib.pyplot as plt
        
        # Use smaller parameters for speed
        ofdm = fv.OFDM(N_subcarriers=64)
        fbmc = fv.FBMC(N_subcarriers=64, K_overlapping=4)
        
        # Generate small test
        symbols = ofdm.generate_qam_symbols(M=16, num_symbols=5)
        
        ofdm_signal = ofdm.modulate(symbols)
        fbmc_signal = fbmc.modulate(symbols)
        
        # Quick spectral analysis
        ofdm_spectrum = ofdm.compute_spectrum(ofdm_signal, nfft=256)
        fbmc_spectrum = fbmc.compute_spectrum(fbmc_signal, nfft=256)
        
        # Verify spectra are different (basic sanity check)
        import numpy as np
        if not np.array_equal(ofdm_spectrum, fbmc_spectrum):
            print("   ✓ OFDM and FBMC produce different spectra")
        else:
            print("   ⚠️  OFDM and FBMC spectra are identical (unexpected)")
        
        # Test basic plotting (without display)
        fig, ax = plt.subplots(figsize=(8, 4))
        freq = np.linspace(-0.5, 0.5, len(ofdm_spectrum))
        ax.plot(freq, ofdm_spectrum, label='OFDM')
        ax.plot(freq, fbmc_spectrum, label='FBMC')
        ax.legend()
        ax.set_title('Quick Test Spectrum')
        
        # Save test plot
        test_plot_path = Path("logs") / "test_spectrum.png"
        fig.savefig(test_plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        if test_plot_path.exists():
            print(f"   ✓ Test plot saved: {test_plot_path}")
            # Clean up test file
            test_plot_path.unlink()
        else:
            print("   ⚠️  Test plot not saved")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Quick validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def performance_benchmark():
    """Quick performance benchmark."""
    print("⏱️  Running performance benchmark...")
    
    try:
        import fbmc_ofdm_validation as fv
        import time
        
        # Benchmark parameters
        N = 256
        num_symbols = 10
        
        ofdm = fv.OFDM(N_subcarriers=N)
        fbmc = fv.FBMC(N_subcarriers=N, K_overlapping=4)
        
        symbols = ofdm.generate_qam_symbols(M=16, num_symbols=num_symbols)
        
        # Time OFDM modulation
        start_time = time.time()
        ofdm_signal = ofdm.modulate(symbols)
        ofdm_time = time.time() - start_time
        
        # Time FBMC modulation  
        start_time = time.time()
        fbmc_signal = fbmc.modulate(symbols)
        fbmc_time = time.time() - start_time
        
        print(f"   ✓ OFDM modulation: {ofdm_time*1000:.2f} ms")
        print(f"   ✓ FBMC modulation: {fbmc_time*1000:.2f} ms")
        print(f"   ✓ FBMC/OFDM ratio: {fbmc_time/ofdm_time:.2f}×")
        
        # Time spectrum computation
        start_time = time.time()
        ofdm_spectrum = ofdm.compute_spectrum(ofdm_signal)
        spectrum_time = time.time() - start_time
        print(f"   ✓ Spectrum computation: {spectrum_time*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Performance benchmark failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("FBMC-TTDF Validation Framework - System Check")
    print("=" * 60)
    
    all_passed = True
    
    # Run all tests
    tests = [
        ("Environment Check", check_environment),
        ("Import Check", check_imports),
        ("Directory Structure", test_directory_structure),
        ("Basic Functionality", test_basic_functionality),
        ("Quick Validation", run_quick_validation),
        ("Performance Benchmark", performance_benchmark),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            all_passed = all_passed and result
        except Exception as e:
            print(f"   ✗ {test_name} crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Run 'uv run main.py' for full validation")
        print("2. Check 'logs/' directory for generated plots")
        print("3. See docs/USAGE.md for advanced examples")
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        print("1. Ensure Python 3.12+ is installed")
        print("2. Check that all dependencies are installed")
        print("3. Verify write permissions in the current directory")
        print("4. See docs/INSTALLATION.md for detailed setup")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)