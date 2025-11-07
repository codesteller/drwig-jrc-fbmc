<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI Logo" width="150"/>
  
  # Usage Guide
  
  **FBMC vs OFDM Validation Framework**
</div>

---

This guide provides detailed examples and use cases for the FBMC vs OFDM Validation Framework.

## 🚀 Quick Start

### Basic Usage

The simplest way to run the complete validation:

```bash
cd fbmc_ttdf
uv run main.py
```

This generates all validation plots and metrics in the `logs/` directory.

## 📊 Understanding the Output

### Generated Files

#### 1. Spectral Comparison (`spectral_comparison.png`)
- **Left Plot**: Full spectral comparison showing main lobe and sidelobes
- **Right Plot**: Zoomed view highlighting out-of-band (OOB) emissions
- **Key Insight**: FBMC shows superior spectral containment (10-40 dB better OOB)

#### 2. Range-Doppler Maps (`range_doppler_comparison.png`)
- **Left**: OFDM range-Doppler response
- **Right**: FBMC range-Doppler response  
- **Key Insight**: FBMC shows reduced sidelobes and better target separation

#### 3. Ambiguity Functions (`ambiguity_comparison.png`)
- **Contour plots**: Time-frequency localization characteristics
- **Key Insight**: FBMC provides better "thumbtack" shape for radar applications

#### 4. Doppler Tolerance (`doppler_tolerance.png`)
- **Correlation degradation** vs Doppler frequency
- **3 dB points**: Marked for both waveforms
- **Key Insight**: FBMC typically shows 2× better Doppler tolerance

#### 5. Metrics Table (`metrics_table.png`)
- **Comprehensive comparison** of all performance metrics
- **Color coding**: Green for improvements, red for degradations
- **Key Metrics**: Spectral efficiency, OOB emissions, PAPR, range sidelobes

### Performance Metrics Interpretation

```python
# Example output interpretation
Spectral Efficiency:
   OFDM: 0.800  # Reduced by cyclic prefix overhead
   FBMC: 1.000  # No cyclic prefix needed
   Improvement: 25.0%

Out-of-Band Emissions @ 1.5×BW:
   OFDM: -26.4 dB    # Higher sidelobes
   FBMC: -62.7 dB    # Superior filtering
   Improvement: 36.3 dB

Range Sidelobe Level:
   OFDM: -26.0 dB    # Correlation sidelobes
   FBMC: -41.3 dB    # Better localization
   Improvement: 15.3 dB
```

## 🔧 Customization Examples

### 1. Modifying System Parameters

Edit the main parameters in `fbmc_ofdm_validation.py`:

```python
# Basic parameter modification
class OFDM(WaveformGenerator):
    def __init__(self, N_subcarriers=512, cp_ratio=0.125, sample_rate=2e9):
        # Increased subcarriers, reduced CP, higher sample rate
        super().__init__(N_subcarriers, cp_ratio, sample_rate)

class FBMC(WaveformGenerator):
    def __init__(self, N_subcarriers=512, K_overlapping=6, sample_rate=2e9):
        # Matched subcarriers, higher overlapping factor
        super().__init__(N_subcarriers, 0, sample_rate)
        self.K = K_overlapping
```

### 2. Different Modulation Orders

```python
# Test different QAM orders
def compare_modulation_orders():
    """Compare performance across different modulation orders."""
    orders = [4, 16, 64, 256]
    results = {}
    
    for M in orders:
        ofdm = OFDM()
        fbmc = FBMC()
        
        symbols = ofdm.generate_qam_symbols(M=M, num_symbols=20)
        
        ofdm_signal = ofdm.modulate(symbols)
        fbmc_signal = fbmc.modulate(symbols)
        
        # Your analysis here
        results[M] = analyze_signals(ofdm_signal, fbmc_signal)
    
    return results
```

### 3. Custom Target Scenarios

```python
# Automotive radar scenario
def automotive_scenario():
    """Simulate typical automotive radar scenario."""
    
    # Vehicle speeds: 0, 50, 100 km/h at different ranges
    ranges = [50, 100, 150]  # meters
    speeds = [0, 50, 100]    # km/h
    
    # Convert to delays and Doppler shifts
    c = 3e8  # Speed of light
    fc = 77e9  # 77 GHz automotive radar
    
    delays = [2 * r / c for r in ranges]  # Two-way propagation
    dopplers = [2 * fc * (v/3.6) / c for v in speeds]  # Doppler shift
    amplitudes = [1.0, 0.7, 0.5]  # Different RCS values
    
    return delays, dopplers, amplitudes

# Usage in main code
delays, dopplers, amplitudes = automotive_scenario()
ofdm_rx = radar.add_target_returns(ofdm_signal, delays, dopplers, amplitudes)
```

### 4. Extended Analysis Functions

```python
def plot_ber_comparison():
    """Compare BER performance in AWGN channel."""
    snr_db = np.arange(0, 25, 2)
    ber_ofdm = []
    ber_fbmc = []
    
    for snr in snr_db:
        # Generate signals
        ofdm = OFDM()
        fbmc = FBMC()
        
        # Add noise based on SNR
        noise_power = 10**(-snr/10)
        
        # Simulate transmission and compute BER
        ber_o = simulate_ber(ofdm, noise_power)
        ber_f = simulate_ber(fbmc, noise_power)
        
        ber_ofdm.append(ber_o)
        ber_fbmc.append(ber_f)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db, ber_ofdm, 'b-o', label='OFDM')
    plt.semilogy(snr_db, ber_fbmc, 'r-s', label='FBMC')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('BER Performance Comparison')
    plt.grid(True)
    plt.legend()
    return plt.gcf()
```

## 📈 Advanced Analysis Examples

### 1. Parameter Sweep Study

```python
def parameter_sweep_analysis():
    """Systematic parameter sweep analysis."""
    
    # Parameters to sweep
    N_values = [128, 256, 512]
    K_values = [2, 4, 6, 8]
    cp_ratios = [0.125, 0.25, 0.5]
    
    results = []
    
    for N in N_values:
        for K in K_values:
            for cp in cp_ratios:
                print(f"Testing N={N}, K={K}, CP={cp}")
                
                # Create waveforms
                ofdm = OFDM(N_subcarriers=N, cp_ratio=cp)
                fbmc = FBMC(N_subcarriers=N, K_overlapping=K)
                
                # Generate test signals
                symbols = ofdm.generate_qam_symbols(M=16, num_symbols=10)
                ofdm_sig = ofdm.modulate(symbols)
                fbmc_sig = fbmc.modulate(symbols)
                
                # Compute metrics
                ofdm_papr = compute_papr(ofdm_sig)
                fbmc_papr = compute_papr(fbmc_sig)
                
                ofdm_oob = compute_oob_emissions(ofdm_sig, N)
                fbmc_oob = compute_oob_emissions(fbmc_sig, N)
                
                # Store results
                results.append({
                    'N': N, 'K': K, 'CP': cp,
                    'OFDM_PAPR': ofdm_papr, 'FBMC_PAPR': fbmc_papr,
                    'OFDM_OOB': ofdm_oob, 'FBMC_OOB': fbmc_oob
                })
    
    return results

def plot_parameter_sweep_results(results):
    """Visualize parameter sweep results."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # PAPR vs K
    for N in df['N'].unique():
        data = df[df['N'] == N]
        axes[0,0].plot(data['K'], data['FBMC_PAPR'], 'o-', label=f'N={N}')
    axes[0,0].set_xlabel('Overlapping Factor K')
    axes[0,0].set_ylabel('FBMC PAPR (dB)')
    axes[0,0].set_title('PAPR vs Overlapping Factor')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # OOB vs CP
    for N in df['N'].unique():
        data = df[df['N'] == N]
        axes[0,1].plot(data['CP'], data['OFDM_OOB'], 's-', label=f'N={N}')
    axes[0,1].set_xlabel('Cyclic Prefix Ratio')
    axes[0,1].set_ylabel('OFDM OOB (dB)')
    axes[0,1].set_title('OOB vs Cyclic Prefix')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Continue for other subplots...
    
    plt.tight_layout()
    return fig
```

### 2. Channel Model Integration

```python
def multipath_channel_analysis():
    """Analyze performance in multipath channels."""
    
    # Define channel models
    channels = {
        'AWGN': {'type': 'awgn'},
        'Rayleigh': {'type': 'rayleigh', 'fade_rate': 100},
        'Rician': {'type': 'rician', 'k_factor': 10},
        'Multipath': {
            'type': 'multipath',
            'delays': [0, 1e-6, 2e-6],
            'gains': [1.0, 0.5, 0.25]
        }
    }
    
    results = {}
    
    for channel_name, channel_params in channels.items():
        print(f"Testing {channel_name} channel...")
        
        # Generate signals
        ofdm = OFDM()
        fbmc = FBMC()
        symbols = ofdm.generate_qam_symbols(M=16, num_symbols=50)
        
        ofdm_signal = ofdm.modulate(symbols)
        fbmc_signal = fbmc.modulate(symbols)
        
        # Apply channel
        ofdm_rx = apply_channel(ofdm_signal, channel_params)
        fbmc_rx = apply_channel(fbmc_signal, channel_params)
        
        # Compute performance metrics
        ofdm_mse = compute_mse(ofdm_signal, ofdm_rx)
        fbmc_mse = compute_mse(fbmc_signal, fbmc_rx)
        
        results[channel_name] = {
            'OFDM_MSE': ofdm_mse,
            'FBMC_MSE': fbmc_mse,
            'Improvement': 10*np.log10(ofdm_mse/fbmc_mse)
        }
    
    return results

def apply_channel(signal, channel_params):
    """Apply various channel models."""
    if channel_params['type'] == 'awgn':
        noise = np.random.normal(0, 0.1, len(signal)) + \
                1j * np.random.normal(0, 0.1, len(signal))
        return signal + noise
    
    elif channel_params['type'] == 'multipath':
        output = np.zeros_like(signal, dtype=complex)
        for delay, gain in zip(channel_params['delays'], channel_params['gains']):
            delay_samples = int(delay * 1e9)  # Assuming 1 GHz sample rate
            delayed_signal = np.roll(signal, delay_samples) * gain
            output += delayed_signal
        return output
    
    # Add other channel models as needed
    return signal
```

### 3. Hardware Implementation Analysis

```python
def complexity_analysis():
    """Analyze computational complexity."""
    
    N_values = [64, 128, 256, 512, 1024]
    
    ofdm_flops = []
    fbmc_flops = []
    
    for N in N_values:
        # OFDM complexity: mainly IFFT
        ofdm_ops = N * np.log2(N)  # Complex IFFT
        ofdm_flops.append(ofdm_ops)
        
        # FBMC complexity: polyphase filtering + modulation
        K = 4  # Overlapping factor
        fbmc_ops = N * K * 2 + N * np.log2(N)  # Filtering + DFT
        fbmc_flops.append(fbmc_ops)
    
    # Plot complexity comparison
    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, ofdm_flops, 'b-o', label='OFDM', linewidth=2)
    plt.loglog(N_values, fbmc_flops, 'r-s', label='FBMC', linewidth=2)
    plt.xlabel('Number of Subcarriers')
    plt.ylabel('Operations per Symbol')
    plt.title('Computational Complexity Comparison')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    return plt.gcf()

def memory_analysis():
    """Analyze memory requirements."""
    
    N_values = [64, 128, 256, 512, 1024]
    
    ofdm_memory = []
    fbmc_memory = []
    
    for N in N_values:
        # OFDM memory: input buffer + FFT twiddles
        ofdm_mem = N * 8 + N * 8  # Complex samples (8 bytes each)
        ofdm_memory.append(ofdm_mem / 1024)  # Convert to KB
        
        # FBMC memory: filter coefficients + input buffer
        K = 4
        fbmc_mem = N * K * 8 + N * 8  # Filter + input buffer
        fbmc_memory.append(fbmc_mem / 1024)  # Convert to KB
    
    # Create comparison table
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(N_values))
    width = 0.35
    
    ax.bar(x - width/2, ofdm_memory, width, label='OFDM', color='blue', alpha=0.7)
    ax.bar(x + width/2, fbmc_memory, width, label='FBMC', color='red', alpha=0.7)
    
    ax.set_xlabel('Number of Subcarriers')
    ax.set_ylabel('Memory Requirements (KB)')
    ax.set_title('Memory Usage Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(N_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
```

## 🎯 Use Case Examples

### 1. Academic Paper Validation

```python
def generate_paper_figures():
    """Generate publication-quality figures for academic paper."""
    
    # Set publication parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.figsize': (10, 6),
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    
    # Generate main comparison figures
    fig1 = plot_spectral_comparison()
    fig2 = plot_range_doppler_comparison()
    fig3 = plot_ambiguity_functions()
    fig4 = plot_doppler_tolerance()
    
    # Save in publication format
    fig1.savefig('paper_spectral.pdf', format='pdf', bbox_inches='tight', dpi=300)
    fig2.savefig('paper_range_doppler.pdf', format='pdf', bbox_inches='tight', dpi=300)
    fig3.savefig('paper_ambiguity.pdf', format='pdf', bbox_inches='tight', dpi=300)
    fig4.savefig('paper_doppler.pdf', format='pdf', bbox_inches='tight', dpi=300)
    
    print("Publication figures saved as PDF files")
    
    return [fig1, fig2, fig3, fig4]
```

### 2. System Design Trade-offs

```python
def system_design_analysis():
    """Analyze trade-offs for system design decisions."""
    
    # Define design scenarios
    scenarios = {
        'High Mobility': {
            'max_doppler': 5000,  # Hz
            'requirement': 'doppler_tolerance'
        },
        'Spectrum Constrained': {
            'adjacent_channel_power': -60,  # dB
            'requirement': 'oob_emissions'
        },
        'Power Limited': {
            'max_papr': 8,  # dB
            'requirement': 'papr'
        },
        'Low Complexity': {
            'max_complexity': 1.5,  # Relative to OFDM
            'requirement': 'complexity'
        }
    }
    
    recommendations = {}
    
    for scenario, params in scenarios.items():
        print(f"\nAnalyzing {scenario} scenario...")
        
        # Run simulation
        metrics = compute_performance_metrics()
        
        # Extract relevant metric
        if params['requirement'] == 'doppler_tolerance':
            # Higher is better
            ofdm_val = extract_doppler_3db_point('OFDM')
            fbmc_val = extract_doppler_3db_point('FBMC')
            recommendation = 'FBMC' if fbmc_val > ofdm_val else 'OFDM'
            
        elif params['requirement'] == 'oob_emissions':
            # Lower is better (more negative)
            ofdm_val = extract_oob_emissions('OFDM')
            fbmc_val = extract_oob_emissions('FBMC')
            recommendation = 'FBMC' if fbmc_val < ofdm_val else 'OFDM'
            
        # Add other requirements...
        
        recommendations[scenario] = {
            'recommended_waveform': recommendation,
            'ofdm_performance': ofdm_val,
            'fbmc_performance': fbmc_val,
            'improvement': calculate_improvement(ofdm_val, fbmc_val)
        }
    
    return recommendations
```

### 3. Educational Demonstrations

```python
def create_interactive_demo():
    """Create interactive demonstration for educational use."""
    
    import ipywidgets as widgets
    from IPython.display import display
    
    # Create interactive widgets
    n_slider = widgets.IntSlider(value=256, min=64, max=1024, step=64, 
                                description='Subcarriers:')
    k_slider = widgets.IntSlider(value=4, min=2, max=8, step=1,
                                description='FBMC K:')
    cp_slider = widgets.FloatSlider(value=0.25, min=0.125, max=0.5, step=0.125,
                                   description='OFDM CP:')
    
    def update_plots(N, K, CP):
        """Update plots based on widget values."""
        ofdm = OFDM(N_subcarriers=N, cp_ratio=CP)
        fbmc = FBMC(N_subcarriers=N, K_overlapping=K)
        
        symbols = ofdm.generate_qam_symbols(M=16, num_symbols=10)
        
        ofdm_signal = ofdm.modulate(symbols)
        fbmc_signal = fbmc.modulate(symbols)
        
        # Generate comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Time domain
        ax1.plot(np.real(ofdm_signal[:500]), 'b-', label='OFDM', alpha=0.7)
        ax1.plot(np.real(fbmc_signal[:500]), 'r-', label='FBMC', alpha=0.7)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time Domain Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain
        ofdm_spectrum = np.fft.fftshift(np.fft.fft(ofdm_signal, 2048))
        fbmc_spectrum = np.fft.fftshift(np.fft.fft(fbmc_signal, 2048))
        
        freq = np.linspace(-0.5, 0.5, 2048)
        
        ax2.plot(freq, 20*np.log10(np.abs(ofdm_spectrum)), 'b-', label='OFDM')
        ax2.plot(freq, 20*np.log10(np.abs(fbmc_spectrum)), 'r-', label='FBMC')
        ax2.set_xlabel('Normalized Frequency')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Frequency Domain')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-80, 10])
        
        plt.tight_layout()
        plt.show()
    
    # Create interactive interface
    interactive_plot = widgets.interactive(update_plots, N=n_slider, K=k_slider, CP=cp_slider)
    display(interactive_plot)
```

## 🔍 Troubleshooting Common Issues

### 1. Memory Issues
```python
# Reduce memory usage for large parameter sweeps
def memory_efficient_analysis():
    """Memory-efficient version for large-scale analysis."""
    
    # Process in chunks
    chunk_size = 10
    total_symbols = 100
    
    results = []
    
    for start_idx in range(0, total_symbols, chunk_size):
        end_idx = min(start_idx + chunk_size, total_symbols)
        
        # Process smaller chunks
        symbols_chunk = generate_symbols(chunk_size)
        result_chunk = process_chunk(symbols_chunk)
        results.append(result_chunk)
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return combine_results(results)
```

### 2. Performance Optimization
```python
# Vectorized operations for better performance
def optimized_spectrum_computation(signal, nfft=4096):
    """Optimized spectral computation using vectorization."""
    
    # Use scipy's periodogram for better performance
    from scipy.signal import periodogram
    
    freqs, psd = periodogram(signal, fs=1.0, nfft=nfft, 
                            window='hamming', scaling='density')
    
    psd_db = 10 * np.log10(psd / np.max(psd))
    
    return freqs, psd_db
```

### 3. Numerical Precision
```python
# Handle numerical precision issues
def robust_metric_computation():
    """Robust computation handling edge cases."""
    
    def safe_log10(x, min_val=1e-12):
        """Safely compute log10 avoiding numerical issues."""
        return 10 * np.log10(np.maximum(np.abs(x), min_val))
    
    def safe_divide(numerator, denominator, default_val=0):
        """Safely divide avoiding division by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result[~np.isfinite(result)] = default_val
        return result
```

## 📚 Additional Resources

### Example Notebooks

Create Jupyter notebooks for interactive analysis:

```python
# example_notebook.ipynb
"""
# FBMC vs OFDM Interactive Analysis

This notebook provides interactive exploration of FBMC and OFDM waveforms.

## Setup
"""

import numpy as np
import matplotlib.pyplot as plt
from fbmc_ofdm_validation import OFDM, FBMC, plot_spectral_comparison

# Enable interactive plots
%matplotlib widget

"""
## Basic Comparison
"""

# Generate and compare waveforms
fig = plot_spectral_comparison()
plt.show()

"""
## Parameter Exploration

Use the widgets below to explore different parameter combinations.
"""

# Interactive widgets code here...
```

### Configuration Files

Create YAML configuration for different scenarios:

```yaml
# config/automotive_radar.yaml
scenario: "Automotive Radar 77-81 GHz"

system_parameters:
  N_subcarriers: 256
  sample_rate: 1.0e9
  frequency_band: [77.0e9, 81.0e9]

ofdm_parameters:
  cp_ratio: 0.25
  windowing: "rectangular"

fbmc_parameters:
  K_overlapping: 4
  prototype_filter: "phydyas"

radar_parameters:
  max_range: 200  # meters
  max_velocity: 150  # km/h
  range_resolution: 0.75  # meters
  velocity_resolution: 5  # km/h

target_scenarios:
  - name: "Stationary Vehicle"
    range: 50
    velocity: 0
    rcs: 10
  - name: "Approaching Vehicle"  
    range: 100
    velocity: -80
    rcs: 15
  - name: "Motorcycle"
    range: 75
    velocity: 60
    rcs: 2
```

---

<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI" width="80"/>
  <br>
  <strong>© 2025 Gahan AI Private Limited</strong>
  <br>
  <sub>This comprehensive usage guide provides extensive examples for customizing and extending the FBMC vs OFDM validation framework.</sub>
</div>