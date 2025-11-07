'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-09-09 13:51:49
 # @ Modified time: 2025-11-07 14:05:39
 # @ Description: FBMC vs OFDM Validation for Joint Radar-Communication Systems Validation Framework 
 '''


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fft2, fftshift
from scipy.signal.windows import kaiser
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class WaveformGenerator:
    """Base class for waveform generation"""
    
    def __init__(self, N_subcarriers=256, cp_ratio=0.25, sample_rate=1e9):
        self.N = N_subcarriers
        self.cp_ratio = cp_ratio
        self.fs = sample_rate
        self.delta_f = self.fs / self.N  # Subcarrier spacing
        
    def generate_qam_symbols(self, M=16, num_symbols=10):
        """Generate M-QAM symbols"""
        k = int(np.log2(M))
        constellation = []
        for i in range(int(np.sqrt(M))):
            for j in range(int(np.sqrt(M))):
                real = 2*i - np.sqrt(M) + 1
                imag = 2*j - np.sqrt(M) + 1
                constellation.append(real + 1j*imag)
        constellation = np.array(constellation) / np.sqrt(np.mean(np.abs(constellation)**2))
        
        # Random symbol generation
        symbols = np.random.choice(constellation, size=(self.N, num_symbols))
        return symbols

class OFDM(WaveformGenerator):
    """OFDM waveform generator"""
    
    def modulate(self, symbols):
        """OFDM modulation with CP"""
        num_symbols = symbols.shape[1]
        cp_len = int(self.N * self.cp_ratio)
        samples_per_symbol = self.N + cp_len
        
        time_signal = np.zeros(samples_per_symbol * num_symbols, dtype=complex)
        
        for idx in range(num_symbols):
            # IFFT for OFDM modulation
            time_domain = ifft(symbols[:, idx], self.N) * np.sqrt(self.N)
            
            # Add cyclic prefix
            with_cp = np.concatenate([time_domain[-cp_len:], time_domain])
            
            # Place in output signal
            start_idx = idx * samples_per_symbol
            end_idx = start_idx + samples_per_symbol
            time_signal[start_idx:end_idx] = with_cp
            
        return time_signal
    
    def compute_spectrum(self, signal, nfft=4096):
        """Compute power spectral density"""
        window = np.hamming(len(signal))
        spectrum = fft(signal * window, nfft)
        psd = 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum)))
        return psd
    
class FBMC(WaveformGenerator):
    """FBMC-OQAM waveform generator"""
    
    def __init__(self, N_subcarriers=256, K_overlapping=4, sample_rate=1e9):
        super().__init__(N_subcarriers, 0, sample_rate)  # No CP in FBMC
        self.K = K_overlapping  # Overlapping factor
        self.prototype_filter = self.design_phydyas_filter()
        
    def design_phydyas_filter(self):
        """Design PHYDYAS prototype filter"""
        # PHYDYAS coefficients for K=4
        H = np.array([1, 0.97196, 0.707, 0.235147])
        
        # Build symmetric filter
        L = self.K * self.N
        h = np.zeros(L)
        
        for k in range(self.K):
            h[k*self.N:(k+1)*self.N] = H[k] * np.ones(self.N)
            
        # Normalize
        h = h / np.sqrt(np.sum(h**2))
        
        # Apply Kaiser window for better spectral containment
        window = kaiser(L, beta=3)
        h = h * window
        
        return h
    
    def oqam_preprocessing(self, symbols):
        """Convert QAM to OQAM (staggered real/imaginary)"""
        oqam_symbols = np.zeros((self.N, 2*symbols.shape[1]), dtype=float)
        
        for n in range(symbols.shape[1]):
            for k in range(self.N):
                if k % 2 == 0:  # Even subcarriers
                    oqam_symbols[k, 2*n] = np.real(symbols[k, n])
                    oqam_symbols[k, 2*n+1] = np.imag(symbols[k, n])
                else:  # Odd subcarriers
                    oqam_symbols[k, 2*n] = np.imag(symbols[k, n])
                    oqam_symbols[k, 2*n+1] = np.real(symbols[k, n])
                    
        return oqam_symbols
    
    def modulate(self, symbols):
        """FBMC-OQAM modulation"""
        # OQAM preprocessing
        oqam_symbols = self.oqam_preprocessing(symbols)
        num_oqam_symbols = oqam_symbols.shape[1]
        
        # Synthesis filter bank (simplified implementation)
        L = len(self.prototype_filter)
        output_len = (num_oqam_symbols + self.K - 1) * self.N // 2
        time_signal = np.zeros(output_len, dtype=complex)
        
        for n in range(num_oqam_symbols):
            for k in range(self.N):
                # Phase factor for OQAM
                phase = np.exp(1j * np.pi * k * (n + 0.5 * (k % 2)) / 2)
                
                # Upsampled and filtered symbol
                symbol_contribution = oqam_symbols[k, n] * phase
                
                # Apply prototype filter (simplified - full polyphase implementation needed for efficiency)
                start_idx = n * self.N // 2
                end_idx = min(start_idx + L, output_len)
                filter_end = min(L, output_len - start_idx)
                
                # Modulate to subcarrier frequency
                t = np.arange(filter_end)
                carrier = np.exp(1j * 2 * np.pi * k * t / self.N)
                
                time_signal[start_idx:end_idx] += symbol_contribution * \
                    self.prototype_filter[:filter_end] * carrier
                    
        return time_signal
    
    def compute_spectrum(self, signal, nfft=4096):
        """Compute power spectral density"""
        spectrum = fft(signal, nfft)
        psd = 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum)))
        return psd

class RadarProcessor:
    """Radar signal processing for both OFDM and FBMC"""
    
    def __init__(self, waveform_type='OFDM'):
        self.waveform_type = waveform_type
        
    def add_target_returns(self, signal, delays, dopplers, amplitudes, fs=1e9):
        """Add radar target returns with delay and Doppler"""
        received = np.zeros_like(signal, dtype=complex)
        
        for delay, doppler, amp in zip(delays, dopplers, amplitudes):
            # Apply delay (simplified - integer sample delay)
            delay_samples = int(delay * fs)
            delayed = np.roll(signal, delay_samples)
            
            # Apply Doppler shift
            t = np.arange(len(signal)) / fs
            doppler_shift = np.exp(1j * 2 * np.pi * doppler * t)
            
            received += amp * delayed * doppler_shift
            
        # Add noise
        noise_power = 0.01
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                          1j * np.random.randn(len(signal)))
        received += noise
        
        return received
    
    def compute_range_doppler_map(self, tx_signal, rx_signal, N_range=256, N_doppler=256):
        """Compute 2D range-Doppler map"""
        # Reshape for processing (simplified)
        # In practice, need proper symbol-by-symbol processing
        
        # Matched filtering
        matched = signal.correlate(rx_signal, tx_signal, mode='same')
        
        # Reshape and compute 2D FFT
        # This is simplified - actual implementation needs proper windowing
        num_symbols = N_doppler
        samples_per_symbol = len(matched) // num_symbols
        
        reshaped = matched[:samples_per_symbol * num_symbols].reshape(num_symbols, samples_per_symbol)
        
        # Apply window
        range_window = np.hamming(samples_per_symbol)
        doppler_window = np.hamming(num_symbols)
        windowed = reshaped * doppler_window[:, np.newaxis] * range_window[np.newaxis, :]
        
        # 2D FFT for range-Doppler
        rd_map = fftshift(fft2(windowed, s=(N_range, N_doppler)))
        rd_map_db = 20 * np.log10(np.abs(rd_map) / np.max(np.abs(rd_map)))
        
        return rd_map_db
    
    def compute_ambiguity_function(self, signal, max_delay=100, max_doppler=100):
        """Compute ambiguity function"""
        delays = np.arange(-max_delay, max_delay)
        dopplers = np.linspace(-max_doppler, max_doppler, 201)
        
        ambiguity = np.zeros((len(delays), len(dopplers)), dtype=complex)
        
        for i, tau in enumerate(delays):
            for j, fd in enumerate(dopplers):
                # Shift signal
                shifted = np.roll(signal, tau)
                
                # Apply Doppler
                t = np.arange(len(signal))
                doppler_shift = np.exp(1j * 2 * np.pi * fd * t / len(signal))
                
                # Compute correlation
                ambiguity[i, j] = np.sum(signal * np.conj(shifted * doppler_shift))
                
        ambiguity_db = 20 * np.log10(np.abs(ambiguity) / np.max(np.abs(ambiguity)))
        
        return delays, dopplers, ambiguity_db

def plot_spectral_comparison():
    """Compare spectral properties of OFDM vs FBMC"""
    print("Generating Spectral Comparison...")
    
    # Generate waveforms
    ofdm = OFDM(N_subcarriers=256)
    fbmc = FBMC(N_subcarriers=256, K_overlapping=4)
    
    # Generate symbols
    symbols = ofdm.generate_qam_symbols(M=16, num_symbols=10)
    
    # Modulate
    ofdm_signal = ofdm.modulate(symbols)
    fbmc_signal = fbmc.modulate(symbols)
    
    # Compute spectra
    nfft = 8192
    ofdm_psd = ofdm.compute_spectrum(ofdm_signal, nfft)
    fbmc_psd = fbmc.compute_spectrum(fbmc_signal, nfft)
    
    # Frequency axis
    freq = np.linspace(-ofdm.fs/2, ofdm.fs/2, nfft)
    freq_normalized = freq / ofdm.delta_f  # Normalize to subcarrier spacing
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full spectrum
    ax1.plot(freq_normalized, fftshift(ofdm_psd), 'b-', label='OFDM', linewidth=1.5)
    ax1.plot(freq_normalized, fftshift(fbmc_psd), 'r-', label='FBMC', linewidth=1.5)
    ax1.set_xlabel('Normalized Frequency (f/Δf)')
    ax1.set_ylabel('Power Spectral Density (dB)')
    ax1.set_title('Spectral Comparison: OFDM vs FBMC')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-150, 150])
    ax1.set_ylim([-80, 5])
    ax1.legend()
    
    # Zoomed view for out-of-band emissions
    ax2.plot(freq_normalized, fftshift(ofdm_psd), 'b-', label='OFDM', linewidth=1.5)
    ax2.plot(freq_normalized, fftshift(fbmc_psd), 'r-', label='FBMC', linewidth=1.5)
    ax2.set_xlabel('Normalized Frequency (f/Δf)')
    ax2.set_ylabel('Power Spectral Density (dB)')
    ax2.set_title('Out-of-Band Emissions (Zoomed)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([130, 145])
    ax2.set_ylim([-80, -20])
    ax2.legend()
    
    # Add measurements
    ofdm_oob = np.max(ofdm_psd[nfft//2 + int(1.2*256):])
    fbmc_oob = np.max(fbmc_psd[nfft//2 + int(1.2*256):])
    
    textstr = f'OOB @ 1.2×BW:\nOFDM: {ofdm_oob:.1f} dB\nFBMC: {fbmc_oob:.1f} dB\nImprovement: {ofdm_oob-fbmc_oob:.1f} dB'
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def plot_range_doppler_comparison():
    """Compare range-Doppler maps"""
    print("Generating Range-Doppler Comparison...")
    
    # Generate waveforms
    ofdm = OFDM(N_subcarriers=256)
    fbmc = FBMC(N_subcarriers=256, K_overlapping=4)
    
    # Generate symbols
    symbols = ofdm.generate_qam_symbols(M=16, num_symbols=32)
    
    # Modulate
    ofdm_signal = ofdm.modulate(symbols)
    fbmc_signal = fbmc.modulate(symbols)
    
    # Radar processor
    radar = RadarProcessor()
    
    # Add targets (delay in seconds, Doppler in Hz, amplitude)
    delays = [1e-6, 3e-6, 5e-6]
    dopplers = [1000, -2000, 500]
    amplitudes = [1.0, 0.7, 0.5]
    
    ofdm_rx = radar.add_target_returns(ofdm_signal, delays, dopplers, amplitudes)
    fbmc_rx = radar.add_target_returns(fbmc_signal, delays, dopplers, amplitudes)
    
    # Compute range-Doppler maps
    ofdm_rd = radar.compute_range_doppler_map(ofdm_signal, ofdm_rx)
    fbmc_rd = radar.compute_range_doppler_map(fbmc_signal, fbmc_rx)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = ax1.imshow(ofdm_rd, aspect='auto', cmap='jet', vmin=-40, vmax=0,
                     extent=[-128, 128, -128, 128])
    ax1.set_xlabel('Doppler Bins')
    ax1.set_ylabel('Range Bins')
    ax1.set_title('OFDM Range-Doppler Map')
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    im2 = ax2.imshow(fbmc_rd, aspect='auto', cmap='jet', vmin=-40, vmax=0,
                     extent=[-128, 128, -128, 128])
    ax2.set_xlabel('Doppler Bins')
    ax2.set_ylabel('Range Bins')
    ax2.set_title('FBMC Range-Doppler Map')
    plt.colorbar(im2, ax=ax2, label='Magnitude (dB)')
    
    plt.tight_layout()
    return fig

def plot_ambiguity_functions():
    """Compare ambiguity functions"""
    print("Generating Ambiguity Function Comparison...")
    
    # Generate shorter signals for ambiguity function
    ofdm = OFDM(N_subcarriers=64)
    fbmc = FBMC(N_subcarriers=64, K_overlapping=4)
    
    symbols = ofdm.generate_qam_symbols(M=16, num_symbols=4)
    
    ofdm_signal = ofdm.modulate(symbols)
    fbmc_signal = fbmc.modulate(symbols)
    
    radar = RadarProcessor()
    
    # Compute ambiguity functions
    delays_o, doppler_o, ambig_ofdm = radar.compute_ambiguity_function(ofdm_signal[:1000])
    delays_f, doppler_f, ambig_fbmc = radar.compute_ambiguity_function(fbmc_signal[:1000])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = ax1.contourf(doppler_o, delays_o, ambig_ofdm, levels=20, cmap='jet', vmin=-40, vmax=0)
    ax1.set_xlabel('Doppler Shift (normalized)')
    ax1.set_ylabel('Time Delay (samples)')
    ax1.set_title('OFDM Ambiguity Function')
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    im2 = ax2.contourf(doppler_f, delays_f, ambig_fbmc, levels=20, cmap='jet', vmin=-40, vmax=0)
    ax2.set_xlabel('Doppler Shift (normalized)')
    ax2.set_ylabel('Time Delay (samples)')
    ax2.set_title('FBMC Ambiguity Function')
    plt.colorbar(im2, ax=ax2, label='Magnitude (dB)')
    
    plt.tight_layout()
    return fig

def compute_performance_metrics():
    """Compute and compare key performance metrics"""
    print("\nComputing Performance Metrics...")
    print("="*60)
    
    # Initialize waveforms
    ofdm = OFDM(N_subcarriers=256, cp_ratio=0.25)
    fbmc = FBMC(N_subcarriers=256, K_overlapping=4)
    
    # Generate test signals
    symbols = ofdm.generate_qam_symbols(M=64, num_symbols=100)
    ofdm_signal = ofdm.modulate(symbols)
    fbmc_signal = fbmc.modulate(symbols)
    
    # 1. Spectral Efficiency
    ofdm_efficiency = 1 / (1 + ofdm.cp_ratio)  # Account for CP overhead
    fbmc_efficiency = 1.0  # No CP overhead
    improvement_efficiency = (fbmc_efficiency - ofdm_efficiency) / ofdm_efficiency * 100
    
    print(f"1. SPECTRAL EFFICIENCY:")
    print(f"   OFDM: {ofdm_efficiency:.3f}")
    print(f"   FBMC: {fbmc_efficiency:.3f}")
    print(f"   Improvement: {improvement_efficiency:.1f}%")
    
    # 2. Out-of-Band Emissions
    nfft = 8192
    ofdm_psd = ofdm.compute_spectrum(ofdm_signal, nfft)
    fbmc_psd = fbmc.compute_spectrum(fbmc_signal, nfft)
    
    # Measure at 1.5x bandwidth
    oob_idx = int(1.5 * 256)
    ofdm_oob = np.max(ofdm_psd[nfft//2 + oob_idx:nfft//2 + oob_idx + 100])
    fbmc_oob = np.max(fbmc_psd[nfft//2 + oob_idx:nfft//2 + oob_idx + 100])
    
    print(f"\n2. OUT-OF-BAND EMISSIONS @ 1.5×BW:")
    print(f"   OFDM: {ofdm_oob:.1f} dB")
    print(f"   FBMC: {fbmc_oob:.1f} dB")
    print(f"   Improvement: {ofdm_oob - fbmc_oob:.1f} dB")
    
    # 3. PAPR (Peak-to-Average Power Ratio)
    ofdm_papr = 10 * np.log10(np.max(np.abs(ofdm_signal)**2) / np.mean(np.abs(ofdm_signal)**2))
    fbmc_papr = 10 * np.log10(np.max(np.abs(fbmc_signal)**2) / np.mean(np.abs(fbmc_signal)**2))
    
    print(f"\n3. PEAK-TO-AVERAGE POWER RATIO (PAPR):")
    print(f"   OFDM: {ofdm_papr:.2f} dB")
    print(f"   FBMC: {fbmc_papr:.2f} dB")
    print(f"   Difference: {fbmc_papr - ofdm_papr:.2f} dB")
    
    # 4. Range Sidelobe Level (simplified measurement)
    radar = RadarProcessor()
    autocorr_ofdm = np.correlate(ofdm_signal[:1000], ofdm_signal[:1000], mode='same')
    autocorr_fbmc = np.correlate(fbmc_signal[:1000], fbmc_signal[:1000], mode='same')
    
    autocorr_ofdm_db = 20 * np.log10(np.abs(autocorr_ofdm) / np.max(np.abs(autocorr_ofdm)))
    autocorr_fbmc_db = 20 * np.log10(np.abs(autocorr_fbmc) / np.max(np.abs(autocorr_fbmc)))
    
    # Find first sidelobe
    center = len(autocorr_ofdm_db) // 2
    ofdm_sidelobe = np.max(autocorr_ofdm_db[center+20:center+100])
    fbmc_sidelobe = np.max(autocorr_fbmc_db[center+20:center+100])
    
    print(f"\n4. RANGE SIDELOBE LEVEL:")
    print(f"   OFDM: {ofdm_sidelobe:.1f} dB")
    print(f"   FBMC: {fbmc_sidelobe:.1f} dB")
    print(f"   Improvement: {ofdm_sidelobe - fbmc_sidelobe:.1f} dB")
    
    # 5. Computational Complexity (relative)
    ofdm_complexity = 1.0  # Baseline
    fbmc_complexity = 2.0  # Due to polyphase filtering
    
    print(f"\n5. COMPUTATIONAL COMPLEXITY (relative):")
    print(f"   OFDM: {ofdm_complexity:.1f}×")
    print(f"   FBMC: {fbmc_complexity:.1f}×")
    
    print("\n" + "="*60)
    
    # Create summary table
    metrics = {
        'Metric': ['Spectral Efficiency', 'OOB @ 1.5×BW (dB)', 'PAPR (dB)', 
                  'Range Sidelobe (dB)', 'Complexity'],
        'OFDM': [f'{ofdm_efficiency:.3f}', f'{ofdm_oob:.1f}', f'{ofdm_papr:.2f}',
                f'{ofdm_sidelobe:.1f}', f'{ofdm_complexity:.1f}×'],
        'FBMC': [f'{fbmc_efficiency:.3f}', f'{fbmc_oob:.1f}', f'{fbmc_papr:.2f}',
                f'{fbmc_sidelobe:.1f}', f'{fbmc_complexity:.1f}×'],
        'Improvement': [f'{improvement_efficiency:.1f}%', f'{ofdm_oob-fbmc_oob:.1f}',
                       f'{fbmc_papr-ofdm_papr:+.2f}', f'{ofdm_sidelobe-fbmc_sidelobe:.1f}',
                       f'+{(fbmc_complexity-ofdm_complexity)*100:.0f}%']
    }
    
    return metrics

def plot_doppler_tolerance():
    """Analyze Doppler tolerance of both waveforms"""
    print("Analyzing Doppler Tolerance...")
    
    # Test different Doppler values
    doppler_values = np.linspace(0, 5000, 50)  # Hz
    
    ofdm = OFDM(N_subcarriers=256)
    fbmc = FBMC(N_subcarriers=256, K_overlapping=4)
    
    symbols = ofdm.generate_qam_symbols(M=16, num_symbols=10)
    
    ofdm_signal = ofdm.modulate(symbols)
    fbmc_signal = fbmc.modulate(symbols)
    
    ofdm_degradation = []
    fbmc_degradation = []
    
    radar = RadarProcessor()
    
    for doppler in doppler_values:
        # Add single target with varying Doppler
        ofdm_rx = radar.add_target_returns(ofdm_signal, [1e-6], [doppler], [1.0])
        fbmc_rx = radar.add_target_returns(fbmc_signal, [1e-6], [doppler], [1.0])
        
        # Measure correlation peak degradation
        ofdm_corr = np.max(np.abs(np.correlate(ofdm_rx[:1000], ofdm_signal[:1000], mode='valid')))
        fbmc_corr = np.max(np.abs(np.correlate(fbmc_rx[:1000], fbmc_signal[:1000], mode='valid')))
        
        ofdm_degradation.append(ofdm_corr)
        fbmc_degradation.append(fbmc_corr)
    
    # Normalize
    ofdm_degradation = np.array(ofdm_degradation) / ofdm_degradation[0]
    fbmc_degradation = np.array(fbmc_degradation) / fbmc_degradation[0]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(doppler_values/1000, 20*np.log10(ofdm_degradation), 'b-', 
            label='OFDM', linewidth=2, marker='o', markersize=4, markevery=5)
    ax.plot(doppler_values/1000, 20*np.log10(fbmc_degradation), 'r-', 
            label='FBMC', linewidth=2, marker='s', markersize=4, markevery=5)
    
    ax.set_xlabel('Doppler Frequency (kHz)')
    ax.set_ylabel('Correlation Peak Degradation (dB)')
    ax.set_title('Doppler Tolerance Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Find 3dB degradation points
    ofdm_3db = doppler_values[np.argmax(20*np.log10(ofdm_degradation) < -3)]
    fbmc_3db = doppler_values[np.argmax(20*np.log10(fbmc_degradation) < -3)]
    
    ax.axhline(y=-3, color='k', linestyle='--', alpha=0.5, label='3 dB threshold')
    ax.axvline(x=ofdm_3db/1000, color='b', linestyle=':', alpha=0.5)
    ax.axvline(x=fbmc_3db/1000, color='r', linestyle=':', alpha=0.5)
    
    textstr = f'3 dB Doppler Tolerance:\nOFDM: {ofdm_3db:.0f} Hz\nFBMC: {fbmc_3db:.0f} Hz\nRatio: {fbmc_3db/ofdm_3db:.2f}×'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


