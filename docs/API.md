<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI Logo" width="150"/>
  
  # API Reference
  
  **FBMC vs OFDM Validation Framework**
</div>

---

## Core Classes

### WaveformGenerator

Base class for waveform generation with common functionality.

```python
class WaveformGenerator:
    def __init__(self, N_subcarriers=256, cp_ratio=0.25, sample_rate=1e9)
```

**Parameters:**
- `N_subcarriers` (int): Number of subcarriers (default: 256)
- `cp_ratio` (float): Cyclic prefix ratio (default: 0.25)
- `sample_rate` (float): Sample rate in Hz (default: 1e9)

**Methods:**

#### generate_qam_symbols(M=16, num_symbols=10)
Generate M-QAM symbol sequences.

**Parameters:**
- `M` (int): QAM constellation size (4, 16, 64, 256)
- `num_symbols` (int): Number of symbol periods

**Returns:**
- `numpy.ndarray`: Complex symbol matrix of shape (N_subcarriers, num_symbols)

### OFDM

OFDM waveform implementation inheriting from WaveformGenerator.

```python
class OFDM(WaveformGenerator):
    def __init__(self, N_subcarriers=256, cp_ratio=0.25, sample_rate=1e9)
```

**Methods:**

#### modulate(symbols)
Perform OFDM modulation with cyclic prefix insertion.

**Parameters:**
- `symbols` (numpy.ndarray): QAM symbols matrix

**Returns:**
- `numpy.ndarray`: Time-domain OFDM signal

#### compute_spectrum(signal, nfft=4096)
Compute power spectral density using windowed FFT.

**Parameters:**
- `signal` (numpy.ndarray): Time-domain signal
- `nfft` (int): FFT size for spectral analysis

**Returns:**
- `numpy.ndarray`: Power spectral density in dB

### FBMC

FBMC-OQAM waveform implementation with polyphase filtering.

```python
class FBMC(WaveformGenerator):
    def __init__(self, N_subcarriers=256, K_overlapping=4, sample_rate=1e9)
```

**Parameters:**
- `N_subcarriers` (int): Number of subcarriers
- `K_overlapping` (int): Overlapping factor (default: 4)
- `sample_rate` (float): Sample rate in Hz

**Methods:**

#### design_phydyas_filter()
Design PHYDYAS prototype filter for FBMC.

**Returns:**
- `numpy.ndarray`: Prototype filter coefficients

#### oqam_preprocessing(symbols)
Convert QAM symbols to OQAM (staggered real/imaginary).

**Parameters:**
- `symbols` (numpy.ndarray): QAM symbols matrix

**Returns:**
- `numpy.ndarray`: OQAM symbols matrix

#### modulate(symbols)
Perform FBMC-OQAM modulation with synthesis filterbank.

**Parameters:**
- `symbols` (numpy.ndarray): QAM symbols matrix

**Returns:**
- `numpy.ndarray`: Time-domain FBMC signal

### RadarProcessor

Radar signal processing functionality for both waveforms.

```python
class RadarProcessor:
    def __init__(self, waveform_type='OFDM')
```

**Parameters:**
- `waveform_type` (str): Waveform type identifier

**Methods:**

#### add_target_returns(signal, delays, dopplers, amplitudes, fs=1e9)
Simulate radar target returns with delay and Doppler effects.

**Parameters:**
- `signal` (numpy.ndarray): Transmit signal
- `delays` (list): Target delays in seconds
- `dopplers` (list): Doppler shifts in Hz
- `amplitudes` (list): Target amplitudes
- `fs` (float): Sample rate

**Returns:**
- `numpy.ndarray`: Received signal with target returns and noise

#### compute_range_doppler_map(tx_signal, rx_signal, N_range=256, N_doppler=256)
Compute 2D range-Doppler map via matched filtering and 2D FFT.

**Parameters:**
- `tx_signal` (numpy.ndarray): Transmitted waveform
- `rx_signal` (numpy.ndarray): Received signal
- `N_range` (int): Range bins
- `N_doppler` (int): Doppler bins

**Returns:**
- `numpy.ndarray`: Range-Doppler map in dB

#### compute_ambiguity_function(signal, max_delay=100, max_doppler=100)
Compute waveform ambiguity function.

**Parameters:**
- `signal` (numpy.ndarray): Waveform signal
- `max_delay` (int): Maximum delay in samples
- `max_doppler` (int): Maximum Doppler shift

**Returns:**
- `tuple`: (delays, dopplers, ambiguity_db)

## Analysis Functions

### plot_spectral_comparison()
Generate spectral comparison plots for OFDM vs FBMC.

**Returns:**
- `matplotlib.figure.Figure`: Spectral comparison figure

### plot_range_doppler_comparison()
Generate range-Doppler map comparisons.

**Returns:**
- `matplotlib.figure.Figure`: Range-Doppler comparison figure

### plot_ambiguity_functions()
Generate ambiguity function comparison plots.

**Returns:**
- `matplotlib.figure.Figure`: Ambiguity function comparison figure

### plot_doppler_tolerance()
Analyze and plot Doppler tolerance characteristics.

**Returns:**
- `matplotlib.figure.Figure`: Doppler tolerance comparison figure

### compute_performance_metrics()
Compute comprehensive performance metrics comparison.

**Returns:**
- `dict`: Performance metrics dictionary with:
  - `'Metric'`: List of metric names
  - `'OFDM'`: OFDM performance values
  - `'FBMC'`: FBMC performance values
  - `'Improvement'`: Improvement values

## Configuration Parameters

### Default System Parameters

```python
# Waveform Parameters
N_SUBCARRIERS = 256        # Number of subcarriers
SAMPLE_RATE = 1e9          # Sample rate (Hz)
CP_RATIO = 0.25           # OFDM cyclic prefix ratio
K_OVERLAPPING = 4         # FBMC overlapping factor

# Modulation Parameters
QAM_ORDER = 16            # QAM constellation size
NUM_SYMBOLS = 10          # Number of symbol periods

# Radar Parameters
TARGET_DELAYS = [1e-6, 3e-6, 5e-6]      # Target delays (s)
TARGET_DOPPLERS = [1000, -2000, 500]     # Doppler shifts (Hz)
TARGET_AMPLITUDES = [1.0, 0.7, 0.5]      # Target RCS

# Analysis Parameters
NFFT_SPECTRUM = 8192      # FFT size for spectral analysis
N_RANGE_BINS = 256        # Range processing bins
N_DOPPLER_BINS = 256      # Doppler processing bins
```

### Plot Parameters

```python
# Publication-quality settings
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)
```

## Error Handling

The framework includes robust error handling for:

- Invalid parameter ranges
- Memory allocation issues
- Numerical precision problems
- Plot generation failures

## Performance Considerations

- **Memory Usage**: Scales with `N_subcarriers × num_symbols`
- **Computation Time**: FBMC ~2× slower than OFDM due to filtering
- **Accuracy**: Double precision (float64) used throughout
- **Parallelization**: Single-threaded implementation (can be extended)

## Extension Points

### Custom Waveforms

Extend `WaveformGenerator` for new waveform types:

```python
class CustomWaveform(WaveformGenerator):
    def modulate(self, symbols):
        # Your modulation implementation
        pass
```

### Additional Metrics

Add to `compute_performance_metrics()`:

```python
def compute_custom_metric():
    # Custom analysis
    return metric_value
```

### Custom Plots

Follow the existing pattern:

```python
def plot_custom_analysis():
    fig, ax = plt.subplots()
    # Your plotting code
    return fig
```

---

<div align="center">
  <img src="internal/gahan_logo.png" alt="Gahan AI" width="80"/>
  <br>
  <strong>© 2025 Gahan AI Private Limited</strong>
</div>