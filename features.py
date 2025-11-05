# features.py
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_ROOT = Path("data/audio")
FEAT_ROOT = Path("data/features")


def spectral_subtraction(s, noise_frames=6, alpha=1.5, beta=0.01):
    """
    Simple spectral subtraction (operated on magnitude spectrum)
    S: magnitude spectrum (freq_bins, frames)
    noise_frames: use the first few frames to estimate the noise spectrum
    alpha: over-subtraction factor
    beta: spectral floor factor
    """
    # Estimate average noise magnitude spectrum using the first noise_frames frames
    noise_mag = np.mean(s[:, :min(noise_frames, s.shape[1])], axis=1, keepdims=True)
    # Spectral subtraction
    s_clean = s - alpha * noise_mag
    s_clean = np.maximum(s_clean, beta * s)  # prevent negative spectrum, apply spectral floor
    return s_clean


def frame_signal(x, fs=16000, win_ms=25, hop_ms=5):
    # Speech framing: 25 ms window, 10 ms hop (for MFCC)
    win = int(fs * win_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    n_frames = 1 + max(0, (len(x) - win) // hop)
    frames = np.stack([x[i * hop: i * hop + win] for i in range(n_frames)], axis=1)  # shape (win, T)
    return frames


def mag_phase(speech_frame):
    # Hamming window + FFT -> magnitude & phase (Lab 2)
    windowed = speech_frame * np.hamming(len(speech_frame))
    x = np.fft.rfft(windowed, n=512)  # 512-point FFT
    mag = np.abs(x)
    phase = np.angle(x)
    return mag, phase


def mel_filterbank(n_fft=512, sr=16000, n_mels=20, fmin=0, fmax=None):
    # Build triangular Mel filterbank matrix H (shape: n_mels × (n_fft/2+1)) (Lab 3)
    if fmax is None: fmax = sr / 2

    # Hz <-> Mel conversions (Lab 3)
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mmin, mmax = hz2mel(fmin), hz2mel(fmax)
    m_points = np.linspace(mmin, mmax, n_mels + 2)  # include edges
    f_points = mel2hz(m_points)
    bins = np.floor((n_fft // 2 + 1) * f_points / (sr / 2)).astype(int)  # Hz -> spectrum bin

    H = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        if f_center == f_left: f_center += 1  # avoid zero width
        if f_right == f_center: f_right += 1
        # left slope
        H[m - 1, f_left:f_center] = (np.arange(f_left, f_center) - f_left) / (f_center - f_left)
        # right slope
        H[m - 1, f_center:f_right] = (f_right - np.arange(f_center, f_right)) / (f_right - f_center)
    return H


_MEL_H = mel_filterbank()


def dct(x, axis=0):
    """
    DCT-II (orthonormal). Equivalent to librosa/scipy dct(type=2, norm='ortho').
    """
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[axis]
    # Build DCT-II basis
    n = np.arange(N)
    k = n[:, None]
    basis = np.cos(np.pi * (n + 0.5) * k / N)  # [N, N]
    # Orthonormal scaling
    basis *= np.sqrt(2.0 / N)
    basis[0, :] *= 1.0 / np.sqrt(2.0)

    # Move target axis to last, apply tensordot, then move back
    x_move = np.moveaxis(x, axis, -1)  # [..., N]
    y = np.tensordot(x_move, basis, axes=([-1], [0]))  # [..., N]
    y = np.moveaxis(y, -1, axis)
    return y


def wav_to_mfcc(wav_path, sr=16000, add_energy=True, add_deltas=True, denoise=False, alpha=1.5, beta=0.01,
                noise_frames=6):
    x, fs = sf.read(wav_path, dtype='float32')
    assert fs == sr, f"Sample rate mismatch: {fs} != {sr}"
    if x.ndim > 1: x = x[:, 0]
    frames = frame_signal(x, fs=sr, win_ms=25, hop_ms=10)

    mags = []
    for i in range(frames.shape[1]):
        mag, _ = mag_phase(frames[:, i])
        mags.append(mag)
    S = np.stack(mags, axis=1)  # shape (257, T)

    # Front-end denoising
    if denoise:
        S = spectral_subtraction(S, noise_frames=6, alpha=1.5, beta=0.01)

    # Mel filterbank energies via matrix multiplication (Lab 3)
    melE = np.matmul(_MEL_H, S) + 1e-10
    log_mel = np.log(melE)
    # DCT -> MFCC (keep first 13)
    mfcc = dct(log_mel, axis=0)[:13, :]
    feats = mfcc

    if add_energy:
        log_energy = np.log((frames ** 2).sum(axis=0) + 1e-10)[None, :]
        feats = np.concatenate([feats, log_energy], axis=0)

    if add_deltas:
        def delta(F):
            # Simple ±1 frame central difference
            pad = np.pad(F, ((0, 0), (1, 1)), mode='edge')
            return (pad[:, 2:] - pad[:, :-2]) * 0.5

        d = delta(feats)
        dd = delta(d)
        feats = np.concatenate([feats, d, dd], axis=0)

    return feats.astype(np.float32)  # shape: [D, T]


def main(audio_root: Path, sr: int = 16000, denoise=False, alpha=1.5, beta=0.01, noise_frames=6):
    FEAT_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect all wav files
    wavs = sorted(audio_root.rglob("*.wav"))
    if not wavs:
        print(f"No .wav files found under {audio_root}.")
        return

    for i, wav_path in enumerate(wavs, 1):
        label = wav_path.parent.name.lower()
        out_dir = FEAT_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (wav_path.stem + ".npy")

        feats = wav_to_mfcc(wav_path.as_posix(), sr=sr, add_energy=True, add_deltas=True,
                            denoise=denoise, alpha=alpha, beta=beta, noise_frames=noise_frames)
        np.save(out_path, feats)

        # Simple progress display
        if i % 20 == 0 or i == len(wavs):
            print(f"[{i}/{len(wavs)}] {wav_path.name}")


if __name__ == "__main__":
    DENOISE = False  # Whether to enable spectral subtraction
    ALPHA = 2.0  # Over-subtraction factor: 1.5, 2.0
    BETA = 0.05  # Spectral floor factor: 0.01, 0.05
    NOISE_FRAMES = 6  # Number of frames for noise estimation

    main(AUDIO_ROOT, sr=16000, denoise=DENOISE, alpha=ALPHA, beta=BETA, noise_frames=NOISE_FRAMES)
