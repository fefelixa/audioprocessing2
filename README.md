# CMP-6026A/CMP-7016A CW1 —— Speaker-dependent（20 names）isolated word recognition

cording to the Lab 1–5 and CW1 to plan week coding：

- Week 1（Lab 1）`week1_io/io_and_plots.py`

  - Recording, playback, save/load WAV, waveform and narrow-/wide-band spectrogram display。

- Week 2（Lab 2）`week2_stft/stft.py`

  - `magAndPhase`，framing and whole-clip STFT magnitude half-spectrum extraction and visualization。

- Week 3（Lab 3）`week3_filterbank/filterbanks_mfcc.py`

  - Linear rectangular filterbank（sum/basis vectors）、Mel triangular filterbank、MFCC（with optional Δ/ΔΔ）。

- Week 4（Lab 4）`week4_model/cnn_model.py`

  - Load `.npy` features，padding，one-hot，splitting，2D-CNN training/validation/testing，confusion matrix。

- Week 5（Lab 5）`week5_noise/noise.py`

  - Target SNR noise mixing，useful for training matched models or conducting noise robustness tests。

## Data Organization & Quick Start

- Put the recordings of the 20 names into `data/audio`，sample rate 16 kHz（Lab 1）。

- Naming suggestion：`<name>_<speaker>_<idx>.wav`，e.g., `alice_me_001.wav`；or place wavs into the `data/audio/<name>/` folder。

- Extract features：

  ```bash

  ```

- Train & evaluate：

  ```bash
  
  ```

- sample inference（weights required first）：

  ```bash
  
  ```

## Tips


## pip

* numpy==1.26.4
* scipy==1.11.4
* matplotlib==3.8.4
* sounddevice==0.4.6
* soundfile==0.12.1
* scikit-learn==1.4.2
* keras==3.3.3
* tensorflow==2.16.1
