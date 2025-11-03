import numpy as np
import soundfile as sf
from pathlib import Path
import random


SR = 16000
CLEAN_ROOT = Path("data/audio")          # clean speech data/audio/<label>/*.wav
NOISE_DIR  = Path("data/noises")         # noise library: .wav
OUT_ROOT   = Path("data/noises")         # output root directory: data_noisy/20dB/<label>/*.wav etc.
SNR_LIST   = [30, 20, 10]

def mix_snr(x, d, snr_db):
    # Lab5: SNR formula and power estimation (mean square)
    Px = np.mean(x**2)
    Pd = np.mean(d**2)
    target_ratio = 10**(-snr_db/10)   # Pd_scaled / Px
    a = np.sqrt(target_ratio * Px / (Pd + 1e-12))
    y = x + a*d
    # Normalize to avoid clipping
    y = 0.99 * y / max(1e-9, np.max(np.abs(y)))
    return y

# Batch process clean speech directory to generate 30/20/10 dB variants
def read_wav(p):
    x, fs = sf.read(p, dtype="float32")
    if x.ndim > 1: x = x[:,0]
    assert fs == SR, f"Sample rate mismatch: {fs} != {SR}"
    # Slight normalization (optional)
    x = x / max(1e-9, np.abs(x).max())
    return x

def random_noise_segment(noise_files, length):
    """Randomly select a segment from the noise library with the same length as the speech;
    if insufficient, loop and concatenate."""
    n = []
    while sum(len(seg) for seg in n) < length:
        p = random.choice(noise_files)
        d = read_wav(p)
        n.append(d)
    dcat = np.concatenate(n)
    start = np.random.randint(0, len(dcat) - length + 1)
    return dcat[start:start+length]

def main():
    noise_files = sorted(NOISE_DIR.glob("*.wav"))
    assert noise_files, f"No noise wavs found in {NOISE_DIR}"

    for wav in sorted(CLEAN_ROOT.rglob("*.wav")):
        x = read_wav(wav)
        for snr in SNR_LIST:
            d = random_noise_segment(noise_files, len(x))
            y = mix_snr(x, d, snr)
            rel = wav.relative_to(CLEAN_ROOT)  # <label>/file.wav
            out_dir = OUT_ROOT / f"{snr}dB" / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / rel.name
            sf.write(out_path.as_posix(), y, SR)
            print(f"SAVED: {out_path}")

if __name__ == "__main__":
    main()
