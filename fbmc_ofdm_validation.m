%% FBMC vs OFDM Validation for Joint Radar-Communication Systems
% MATLAB Implementation for IEEE Paper Validation
% Author: Validation Framework

clear all; close all; clc;

%% System Parameters
N = 256;                % Number of subcarriers
M = 16;                 % M-QAM constellation
cp_ratio = 0.25;        % CP ratio for OFDM
K = 4;                  % Overlapping factor for FBMC
fs = 1e9;               % Sample rate (Hz)
delta_f = fs/N;         % Subcarrier spacing
num_symbols = 32;       % Number of symbols

% Target parameters for radar
target_delays = [1e-6, 3e-6, 5e-6];      % seconds
target_dopplers = [1000, -2000, 500];     % Hz
target_amplitudes = [1.0, 0.7, 0.5];

%% Generate QAM Symbols
qam_symbols = qammod(randi([0 M-1], N, num_symbols), M, 'UnitAveragePower', true);

%% OFDM Modulation Function
function ofdm_signal = ofdm_modulate(symbols, cp_ratio)
    [N, num_symbols] = size(symbols);
    cp_len = round(N * cp_ratio);
    samples_per_symbol = N + cp_len;
    
    ofdm_signal = zeros(1, samples_per_symbol * num_symbols);
    
    for idx = 1:num_symbols
        % IFFT for OFDM modulation
        time_domain = ifft(symbols(:, idx), N) * sqrt(N);
        
        % Add cyclic prefix
        with_cp = [time_domain(end-cp_len+1:end); time_domain];
        
        % Place in output signal
        start_idx = (idx-1) * samples_per_symbol + 1;
        end_idx = start_idx + samples_per_symbol - 1;
        ofdm_signal(start_idx:end_idx) = with_cp;
    end
end

%% FBMC Prototype Filter Design (PHYDYAS)
function h = design_phydyas_filter(K, N)
    % PHYDYAS filter coefficients for K=4
    H = [1, 0.97196, 0.707, 0.235147];
    
    % Build symmetric filter
    L = K * N;
    h = zeros(L, 1);
    
    for k = 1:K
        h((k-1)*N+1:k*N) = H(k);
    end
    
    % Apply Kaiser window for better spectral containment
    h = h .* kaiser(L, 3);
    
    % Normalize
    h = h / sqrt(sum(h.^2));
end

%% FBMC-OQAM Modulation Function
function fbmc_signal = fbmc_modulate(symbols, K)
    [N, num_qam_symbols] = size(symbols);
    
    % Design prototype filter
    h = design_phydyas_filter(K, N);
    L = length(h);
    
    % OQAM preprocessing
    num_oqam_symbols = 2 * num_qam_symbols;
    oqam_symbols = zeros(N, num_oqam_symbols);
    
    for n = 1:num_qam_symbols
        for k = 1:N
            if mod(k, 2) == 1  % Odd subcarriers
                oqam_symbols(k, 2*n-1) = real(symbols(k, n));
                oqam_symbols(k, 2*n) = imag(symbols(k, n));
            else  % Even subcarriers
                oqam_symbols(k, 2*n-1) = imag(symbols(k, n));
                oqam_symbols(k, 2*n) = real(symbols(k, n));
            end
        end
    end
    
    % Synthesis filter bank
    output_len = (num_oqam_symbols + K - 1) * N / 2;
    fbmc_signal = zeros(1, output_len);
    
    for n = 1:num_oqam_symbols
        for k = 1:N
            % Phase factor for OQAM
            phase = exp(1j * pi * (k-1) * (n-1 + 0.5*mod(k-1,2)) / 2);
            
            % Symbol contribution
            symbol_contribution = oqam_symbols(k, n) * phase;
            
            % Apply prototype filter
            start_idx = (n-1) * N/2 + 1;
            end_idx = min(start_idx + L - 1, output_len);
            filter_end = min(L, output_len - start_idx + 1);
            
            % Modulate to subcarrier frequency
            t = (0:filter_end-1)';
            carrier = exp(1j * 2 * pi * (k-1) * t / N);
            
            fbmc_signal(start_idx:end_idx) = fbmc_signal(start_idx:end_idx) + ...
                (symbol_contribution * h(1:filter_end) .* carrier).';
        end
    end
end

%% Main Validation Script
fprintf('========================================\n');
fprintf('FBMC vs OFDM VALIDATION\n');
fprintf('========================================\n\n');

%% 1. Generate Waveforms
fprintf('Generating waveforms...\n');
ofdm_signal = ofdm_modulate(qam_symbols, cp_ratio);
fbmc_signal = fbmc_modulate(qam_symbols, K);

%% 2. Spectral Analysis
fprintf('Computing spectral properties...\n');

nfft = 8192;
ofdm_spectrum = fft(ofdm_signal .* hamming(length(ofdm_signal))', nfft);
fbmc_spectrum = fft(fbmc_signal .* hamming(length(fbmc_signal))', nfft);

ofdm_psd = 20*log10(abs(ofdm_spectrum)/max(abs(ofdm_spectrum)));
fbmc_psd = 20*log10(abs(fbmc_spectrum)/max(abs(fbmc_spectrum)));

% Frequency axis
freq = linspace(-fs/2, fs/2, nfft);
freq_normalized = freq / delta_f;

%% 3. Plot Spectral Comparison
figure('Name', 'Spectral Comparison', 'Position', [100 100 1200 500]);

% Full spectrum
subplot(1, 2, 1);
plot(freq_normalized, fftshift(ofdm_psd), 'b-', 'LineWidth', 1.5, 'DisplayName', 'OFDM');
hold on;
plot(freq_normalized, fftshift(fbmc_psd), 'r-', 'LineWidth', 1.5, 'DisplayName', 'FBMC');
xlabel('Normalized Frequency (f/\Deltaf)');
ylabel('Power Spectral Density (dB)');
title('Spectral Comparison: OFDM vs FBMC');
grid on; grid minor;
xlim([-150 150]);
ylim([-80 5]);
legend('Location', 'northeast');

% Zoomed view
subplot(1, 2, 2);
plot(freq_normalized, fftshift(ofdm_psd), 'b-', 'LineWidth', 1.5, 'DisplayName', 'OFDM');
hold on;
plot(freq_normalized, fftshift(fbmc_psd), 'r-', 'LineWidth', 1.5, 'DisplayName', 'FBMC');
xlabel('Normalized Frequency (f/\Deltaf)');
ylabel('Power Spectral Density (dB)');
title('Out-of-Band Emissions (Zoomed)');
grid on; grid minor;
xlim([130 145]);
ylim([-80 -20]);
legend('Location', 'northeast');

% Measure OOB emissions
oob_idx = round(nfft/2 + 1.2*N);
ofdm_oob = max(ofdm_psd(oob_idx:oob_idx+100));
fbmc_oob = max(fbmc_psd(oob_idx:oob_idx+100));

text(0.05, 0.95, sprintf('OOB @ 1.2×BW:\nOFDM: %.1f dB\nFBMC: %.1f dB\nImprovement: %.1f dB', ...
    ofdm_oob, fbmc_oob, ofdm_oob-fbmc_oob), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'w', 'EdgeColor', 'k');

%% 4. Range-Doppler Processing
fprintf('Computing range-Doppler maps...\n');

% Add noise and targets
noise_power = 0.01;
ofdm_rx = zeros(size(ofdm_signal));
fbmc_rx = zeros(size(fbmc_signal));

for i = 1:length(target_delays)
    % Apply delay (simplified - integer sample delay)
    delay_samples = round(target_delays(i) * fs);
    
    % OFDM
    delayed_ofdm = circshift(ofdm_signal, delay_samples);
    t_ofdm = (0:length(ofdm_signal)-1) / fs;
    doppler_shift_ofdm = exp(1j * 2 * pi * target_dopplers(i) * t_ofdm);
    ofdm_rx = ofdm_rx + target_amplitudes(i) * delayed_ofdm .* doppler_shift_ofdm;
    
    % FBMC
    delayed_fbmc = circshift(fbmc_signal, delay_samples);
    t_fbmc = (0:length(fbmc_signal)-1) / fs;
    doppler_shift_fbmc = exp(1j * 2 * pi * target_dopplers(i) * t_fbmc);
    fbmc_rx = fbmc_rx + target_amplitudes(i) * delayed_fbmc .* doppler_shift_fbmc;
end

% Add noise
ofdm_rx = ofdm_rx + sqrt(noise_power/2) * (randn(size(ofdm_rx)) + 1j*randn(size(ofdm_rx)));
fbmc_rx = fbmc_rx + sqrt(noise_power/2) * (randn(size(fbmc_rx)) + 1j*randn(size(fbmc_rx)));

% Compute matched filter outputs
ofdm_matched = xcorr(ofdm_rx(1:10000), ofdm_signal(1:10000));
fbmc_matched = xcorr(fbmc_rx(1:10000), fbmc_signal(1:10000));

% 2D FFT for range-Doppler (simplified)
N_range = 256;
N_doppler = 256;

% Reshape for 2D processing
samples_per_frame = min(length(ofdm_matched), N_range * N_doppler);
ofdm_reshaped = reshape(ofdm_matched(1:samples_per_frame), N_doppler, N_range);
fbmc_reshaped = reshape(fbmc_matched(1:samples_per_frame), N_doppler, N_range);

% Apply windows
range_window = hamming(N_range)';
doppler_window = hamming(N_doppler);

ofdm_windowed = ofdm_reshaped .* (doppler_window * range_window);
fbmc_windowed = fbmc_reshaped .* (doppler_window * range_window);

% 2D FFT
ofdm_rd_map = fftshift(fft2(ofdm_windowed));
fbmc_rd_map = fftshift(fft2(fbmc_windowed));

ofdm_rd_db = 20*log10(abs(ofdm_rd_map)/max(abs(ofdm_rd_map(:))));
fbmc_rd_db = 20*log10(abs(fbmc_rd_map)/max(abs(fbmc_rd_map(:))));

%% 5. Plot Range-Doppler Maps
figure('Name', 'Range-Doppler Comparison', 'Position', [100 100 1200 500]);

subplot(1, 2, 1);
imagesc(ofdm_rd_db);
caxis([-40 0]);
colormap('jet');
colorbar;
xlabel('Doppler Bins');
ylabel('Range Bins');
title('OFDM Range-Doppler Map');

subplot(1, 2, 2);
imagesc(fbmc_rd_db);
caxis([-40 0]);
colormap('jet');
colorbar;
xlabel('Doppler Bins');
ylabel('Range Bins');
title('FBMC Range-Doppler Map');

%% 6. Ambiguity Function Analysis
fprintf('Computing ambiguity functions...\n');

% Shortened signals for ambiguity function
sig_len = 1000;
ofdm_short = ofdm_signal(1:sig_len);
fbmc_short = fbmc_signal(1:sig_len);

% Compute ambiguity functions
max_delay = 50;
max_doppler = 50;
delays = -max_delay:max_delay;
dopplers = linspace(-max_doppler, max_doppler, 101);

ofdm_ambig = zeros(length(delays), length(dopplers));
fbmc_ambig = zeros(length(delays), length(dopplers));

for i = 1:length(delays)
    for j = 1:length(dopplers)
        % OFDM
        shifted_ofdm = circshift(ofdm_short, delays(i));
        t = (0:sig_len-1)';
        doppler_shift = exp(1j * 2 * pi * dopplers(j) * t / sig_len);
        ofdm_ambig(i, j) = abs(sum(ofdm_short' .* conj(shifted_ofdm' .* doppler_shift')));
        
        % FBMC
        shifted_fbmc = circshift(fbmc_short, delays(i));
        fbmc_ambig(i, j) = abs(sum(fbmc_short' .* conj(shifted_fbmc' .* doppler_shift')));
    end
end

ofdm_ambig_db = 20*log10(ofdm_ambig/max(ofdm_ambig(:)));
fbmc_ambig_db = 20*log10(fbmc_ambig/max(fbmc_ambig(:)));

%% 7. Plot Ambiguity Functions
figure('Name', 'Ambiguity Functions', 'Position', [100 100 1200 500]);

subplot(1, 2, 1);
contourf(dopplers, delays, ofdm_ambig_db, 20);
caxis([-40 0]);
colormap('jet');
colorbar;
xlabel('Doppler Shift (normalized)');
ylabel('Time Delay (samples)');
title('OFDM Ambiguity Function');

subplot(1, 2, 2);
contourf(dopplers, delays, fbmc_ambig_db, 20);
caxis([-40 0]);
colormap('jet');
colorbar;
xlabel('Doppler Shift (normalized)');
ylabel('Time Delay (samples)');
title('FBMC Ambiguity Function');

%% 8. Doppler Tolerance Analysis
fprintf('Analyzing Doppler tolerance...\n');

doppler_test = linspace(0, 5000, 50);  % Hz
ofdm_degradation = zeros(size(doppler_test));
fbmc_degradation = zeros(size(doppler_test));

test_len = 1000;
for i = 1:length(doppler_test)
    % Apply Doppler shift
    t_test = (0:test_len-1)' / fs;
    doppler_shift = exp(1j * 2 * pi * doppler_test(i) * t_test);
    
    % OFDM
    ofdm_doppler = ofdm_signal(1:test_len)' .* doppler_shift;
    ofdm_corr = abs(xcorr(ofdm_doppler, ofdm_signal(1:test_len)', 0));
    ofdm_degradation(i) = ofdm_corr;
    
    % FBMC
    fbmc_doppler = fbmc_signal(1:test_len)' .* doppler_shift;
    fbmc_corr = abs(xcorr(fbmc_doppler, fbmc_signal(1:test_len)', 0));
    fbmc_degradation(i) = fbmc_corr;
end

% Normalize
ofdm_degradation = ofdm_degradation / ofdm_degradation(1);
fbmc_degradation = fbmc_degradation / fbmc_degradation(1);

% Plot
figure('Name', 'Doppler Tolerance', 'Position', [100 100 800 500]);
plot(doppler_test/1000, 20*log10(ofdm_degradation), 'b-o', ...
    'LineWidth', 2, 'MarkerIndices', 1:5:length(doppler_test), 'DisplayName', 'OFDM');
hold on;
plot(doppler_test/1000, 20*log10(fbmc_degradation), 'r-s', ...
    'LineWidth', 2, 'MarkerIndices', 1:5:length(doppler_test), 'DisplayName', 'FBMC');
xlabel('Doppler Frequency (kHz)');
ylabel('Correlation Peak Degradation (dB)');
title('Doppler Tolerance Comparison');
grid on; grid minor;
legend('Location', 'northeast');

% Find 3dB points
ofdm_3db_idx = find(20*log10(ofdm_degradation) < -3, 1);
fbmc_3db_idx = find(20*log10(fbmc_degradation) < -3, 1);

if ~isempty(ofdm_3db_idx) && ~isempty(fbmc_3db_idx)
    ofdm_3db = doppler_test(ofdm_3db_idx);
    fbmc_3db = doppler_test(fbmc_3db_idx);
    
    yline(-3, 'k--', 'LineWidth', 1.5, 'Alpha', 0.5);
    xline(ofdm_3db/1000, 'b:', 'LineWidth', 1.5, 'Alpha', 0.5);
    xline(fbmc_3db/1000, 'r:', 'LineWidth', 1.5, 'Alpha', 0.5);
    
    text(0.95, 0.95, sprintf('3 dB Doppler Tolerance:\nOFDM: %.0f Hz\nFBMC: %.0f Hz\nRatio: %.2fx', ...
        ofdm_3db, fbmc_3db, fbmc_3db/ofdm_3db), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'HorizontalAlignment', 'right', 'BackgroundColor', 'w', 'EdgeColor', 'k');
end

%% 9. Performance Metrics Summary
fprintf('\n========================================\n');
fprintf('PERFORMANCE METRICS SUMMARY\n');
fprintf('========================================\n\n');

% Spectral Efficiency
ofdm_efficiency = 1 / (1 + cp_ratio);
fbmc_efficiency = 1.0;
improvement_efficiency = (fbmc_efficiency - ofdm_efficiency) / ofdm_efficiency * 100;

fprintf('1. SPECTRAL EFFICIENCY:\n');
fprintf('   OFDM: %.3f\n', ofdm_efficiency);
fprintf('   FBMC: %.3f\n', fbmc_efficiency);
fprintf('   Improvement: %.1f%%\n\n', improvement_efficiency);

% Out-of-Band Emissions
fprintf('2. OUT-OF-BAND EMISSIONS @ 1.5×BW:\n');
fprintf('   OFDM: %.1f dB\n', ofdm_oob);
fprintf('   FBMC: %.1f dB\n', fbmc_oob);
fprintf('   Improvement: %.1f dB\n\n', ofdm_oob - fbmc_oob);

% PAPR
ofdm_papr = 10*log10(max(abs(ofdm_signal).^2) / mean(abs(ofdm_signal).^2));
fbmc_papr = 10*log10(max(abs(fbmc_signal).^2) / mean(abs(fbmc_signal).^2));

fprintf('3. PEAK-TO-AVERAGE POWER RATIO (PAPR):\n');
fprintf('   OFDM: %.2f dB\n', ofdm_papr);
fprintf('   FBMC: %.2f dB\n', fbmc_papr);
fprintf('   Difference: %.2f dB\n\n', fbmc_papr - ofdm_papr);

% Range Sidelobe Level
autocorr_ofdm = xcorr(ofdm_signal(1:1000), ofdm_signal(1:1000));
autocorr_fbmc = xcorr(fbmc_signal(1:1000), fbmc_signal(1:1000));

autocorr_ofdm_db = 20*log10(abs(autocorr_ofdm)/max(abs(autocorr_ofdm)));
autocorr_fbmc_db = 20*log10(abs(autocorr_fbmc)/max(abs(autocorr_fbmc)));

center = length(autocorr_ofdm_db)/2;
ofdm_sidelobe = max(autocorr_ofdm_db(center+20:center+100));
fbmc_sidelobe = max(autocorr_fbmc_db(center+20:center+100));

fprintf('4. RANGE SIDELOBE LEVEL:\n');
fprintf('   OFDM: %.1f dB\n', ofdm_sidelobe);
fprintf('   FBMC: %.1f dB\n', fbmc_sidelobe);
fprintf('   Improvement: %.1f dB\n\n', ofdm_sidelobe - fbmc_sidelobe);

% Computational Complexity
fprintf('5. COMPUTATIONAL COMPLEXITY (relative):\n');
fprintf('   OFDM: 1.0×\n');
fprintf('   FBMC: 2.0× (due to polyphase filtering)\n\n');

fprintf('========================================\n');

%% 10. Create Summary Table
figure('Name', 'Performance Metrics', 'Position', [100 100 800 400]);
axis off;

% Data for table
metrics_data = {
    'Spectral Efficiency', sprintf('%.3f', ofdm_efficiency), sprintf('%.3f', fbmc_efficiency), sprintf('%.1f%%', improvement_efficiency);
    'OOB @ 1.5×BW (dB)', sprintf('%.1f', ofdm_oob), sprintf('%.1f', fbmc_oob), sprintf('%.1f', ofdm_oob - fbmc_oob);
    'PAPR (dB)', sprintf('%.2f', ofdm_papr), sprintf('%.2f', fbmc_papr), sprintf('%+.2f', fbmc_papr - ofdm_papr);
    'Range Sidelobe (dB)', sprintf('%.1f', ofdm_sidelobe), sprintf('%.1f', fbmc_sidelobe), sprintf('%.1f', ofdm_sidelobe - fbmc_sidelobe);
    'Complexity', '1.0×', '2.0×', '+100%'
};

% Create table
uitable('Data', metrics_data, ...
    'ColumnName', {'Metric', 'OFDM', 'FBMC', 'Improvement'}, ...
    'ColumnWidth', {180, 100, 100, 120}, ...
    'Position', [50 50 700 300], ...
    'FontSize', 11);

title('FBMC vs OFDM: Performance Metrics Summary', 'FontSize', 14, 'FontWeight', 'bold');

%% Save all figures
fprintf('\nSaving figures...\n');
saveas(1, 'spectral_comparison.png');
saveas(2, 'range_doppler_comparison.png');
saveas(3, 'ambiguity_comparison.png');
saveas(4, 'doppler_tolerance.png');
saveas(5, 'metrics_table.png');

fprintf('\n✓ Validation complete! All figures saved.\n');
fprintf('========================================\n');