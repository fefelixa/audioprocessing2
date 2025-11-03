import sounddevice as sd
import soundfile as sf
import numpy as np

SECONDS = 3.0
OUT_WAV = "data/noises/noise3.wav"
PEAK = 0.99
FS_TARGET = 16000                    # 16khz

def main():
    sd.default.samplerate = FS_TARGET
    sd.default.channels = 1

    print(f"[Rec] Recording {SECONDS}s @ {FS_TARGET} Hz (mono)…")
    audio = sd.rec(int(SECONDS * FS_TARGET), samplerate=FS_TARGET, channels=1, dtype='float32')
    sd.wait()

    x = audio.squeeze()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = (PEAK / peak) * x

    sf.write(OUT_WAV, x, FS_TARGET)
    print(f"[Save] {OUT_WAV}")
    sd.play(x, samplerate=FS_TARGET); sd.wait()

if __name__ == "__main__":
    main()




