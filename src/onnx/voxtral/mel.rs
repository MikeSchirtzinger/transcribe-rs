//! Whisper-compatible log-mel spectrogram feature extraction for Voxtral.
//!
//! Voxtral uses the Whisper feature extractor (WhisperFeatureExtractor in HF):
//! - 16 kHz mono audio
//! - STFT with n_fft = 400, hop = 160, Hann window
//! - 128 slaney-normalized mel filters, f_min = 0, f_max = sr / 2
//! - log10 of the magnitude-squared spectrogram
//! - Dynamic-range clipped to (max - 8.0), then `(x + 4) / 4`
//! - Padded or truncated to exactly 3000 frames (30 s of audio)
//!
//! The filterbank and normalization differ from the shared `features::compute_mel`
//! implementation (htk-mel, natural log), so this module computes its own.

use std::f32::consts::PI;

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const N_SAMPLES: usize = 480_000; // 30 s × 16 kHz
pub const N_FRAMES: usize = 3000; // N_SAMPLES / HOP_LENGTH

/// Compute a Whisper-compatible log-mel spectrogram shaped `[N_MELS, N_FRAMES]`.
///
/// `samples` is expected at 16 kHz mono in the `[-1, 1]` range. Audio shorter
/// than 30 s is zero-padded; longer audio is truncated.
pub fn log_mel_spectrogram(samples: &[f32]) -> Array2<f32> {
    // Pad/truncate to N_SAMPLES (30 seconds). Whisper pads with zeros and
    // reflects the last frame to cover the trailing HOP_LENGTH / 2 positions,
    // but the ONNX models consume exactly N_FRAMES regardless, so we simply
    // zero-pad or clip.
    let mut padded = vec![0f32; N_SAMPLES + N_FFT];
    let copy_len = samples.len().min(N_SAMPLES);
    padded[..copy_len].copy_from_slice(&samples[..copy_len]);

    let window = hann_window(N_FFT);
    let filterbank = slaney_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE as f32);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let freq_bins = N_FFT / 2 + 1;

    // [freq_bins, N_FRAMES] magnitude-squared spectrogram
    let mut magnitudes = Array2::<f32>::zeros((freq_bins, N_FRAMES));
    let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];

    for frame in 0..N_FRAMES {
        let start = frame * HOP_LENGTH;
        for i in 0..N_FFT {
            buf[i] = Complex::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for (bin, val) in buf.iter().enumerate().take(freq_bins) {
            magnitudes[[bin, frame]] = val.norm_sqr();
        }
    }

    // Mel: [N_MELS, freq_bins] × [freq_bins, N_FRAMES] -> [N_MELS, N_FRAMES]
    let mel = filterbank.dot(&magnitudes);

    // log10(max(mel, 1e-10))
    let mut log_spec = mel.mapv(|v| v.max(1e-10).log10());

    // Dynamic range compression: clamp to (max - 8.0)
    let max_val = log_spec
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let floor = max_val - 8.0;
    log_spec.mapv_inplace(|v| v.max(floor));

    // Normalise: (x + 4) / 4 — Whisper convention.
    log_spec.mapv_inplace(|v| (v + 4.0) / 4.0);
    log_spec
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / size as f32).cos())
        .collect()
}

/// Build a slaney-normalised mel filterbank `[n_mels, n_fft/2 + 1]`.
///
/// Matches `librosa.filters.mel(sr, n_fft, n_mels=n_mels, norm='slaney', htk=False)`,
/// which is the default used by HuggingFace's WhisperFeatureExtractor.
fn slaney_mel_filterbank(n_mels: usize, n_fft: usize, sr: f32) -> Array2<f32> {
    let num_bins = n_fft / 2 + 1;
    let f_max = sr / 2.0;
    let f_min = 0.0f32;

    // Slaney mel scale (HTK=false): linear below 1 kHz, log above.
    let mel_low = hz_to_mel_slaney(f_min);
    let mel_high = hz_to_mel_slaney(f_max);

    let num_points = n_mels + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| mel_low + (mel_high - mel_low) * (i as f32) / (num_points as f32 - 1.0))
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz_slaney(m)).collect();

    // FFT bin centre frequencies.
    let bin_freqs: Vec<f32> = (0..num_bins)
        .map(|k| k as f32 * sr / n_fft as f32)
        .collect();

    let mut banks = Array2::<f32>::zeros((n_mels, num_bins));
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];

        let lower_slope = center - left;
        let upper_slope = right - center;

        // Slaney normalisation: divide each filter by (right - left) / 2 so
        // that a flat spectrum maps to unit energy per mel band.
        let norm = 2.0 / (right - left);

        for (k, &f) in bin_freqs.iter().enumerate() {
            let w = if f > left && f <= center {
                (f - left) / lower_slope
            } else if f > center && f < right {
                (right - f) / upper_slope
            } else {
                0.0
            };
            banks[[m, k]] = w * norm;
        }
    }

    banks
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    let logstep: f32 = (6.4f32).ln() / 27.0;
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    let logstep: f32 = (6.4f32).ln() / 27.0;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    }
}
