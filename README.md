# TWET — Trainable Time-Warping Estimation

A PyTorch implementation of **TWET (Time-Warping Estimation Trainable)** for estimating temporal deformations from non-stationary signals using wavelet representations and deep learning.

TWET combines:

* wavelet time-scale analysis,
* differentiable stationarity criteria,
* dilated convolutional neural networks,
* spline-based temporal deformation estimation.

The method is introduced in:

> **TWET: Trainable Time-Warping Estimation Using Time-Scale Representations**

---

## Overview

Time-warping deformations appear in many real-world signals such as:

* audio recordings,
* Doppler signals,
* biomedical signals,
* bioacoustics.

TWET estimates the temporal deformation derivative

y'(t)

from a single observed signal (y(t)), and reconstructs a stationary version of the signal through de-warping.

The framework is designed to:

* handle rapidly varying deformations,
* improve robustness to noise,
* reduce computational cost,
* enable low-latency inference.

---

# Repository Structure

```text
time-warping-estimation/
│
├── TWET/
│   ├── Model.py
│   ├── Time_Warping_Functions.py
│   ├── Stationarity_Score.py
│   └── ...
│
├── Database/
│   ├── Generate_Database.py
│   ├── dataset.npz
│   └── real_dataset.npz
│
├── Common_Functions/
│   ├── CWT.py
│   └── ...
│
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your_username/TWET.git
cd TWET
```


# Dependencies

Main dependencies:

* Python ≥ 3.9
* PyTorch
* NumPy
* SciPy
* Matplotlib

---

# Method

## 1. Wavelet Transform

The observed signal is transformed into a time-scale representation:

$W_y(s,u)$

using a Continuous Wavelet Transform (CWT).

The magnitude of the wavelet transform is used as network input:

```python
Y_wav = np.abs(Y_wav)
```

---

## 2. Neural Architecture

TWET uses:

* dilated 2D convolutions,
* residual blocks,
* GELU activations,
* layer normalization.

The architecture progressively aggregates temporal information across scales.

---

## 3. Deformation Estimation

The network predicts the temporal deformation derivative:

$\widehat{\gamma}'(t)$

using a spline parameterization ensuring smoothness and positivity.

---

## 4. Stationarity Criterion

Training relies on a differentiable stationarity score computed from the de-warped wavelet representation.

The loss combines:

* stationarity minimization,
* spline smoothness regularization.

---

# Training

## Synthetic Dataset

Synthetic signals are generated from:

* stationary Gaussian processes,
* random temporal deformations.


The script automatically:

* loads the dataset,
* computes wavelet transforms,
* trains the model,
* evaluates performance.

---

# Input Shapes

## Signal

$(B, N)$

* `B`: batch size
* `N`: number of samples

## Wavelet Representation

(B, C, F, T)

where:

* `C`: number of wavelet channels,
* `F`: number of frequency bins,
* `T`: number of temporal frames.

---

# Outputs

The model returns:


outputs = {
    "x_pred": x_pred,
    "gamma_prime_pred": gamma_prime_pred,
    "x_wav_pred": x_wav_pred
}

## Estimated stationary signal

```python
x_pred
```

## Estimated deformation derivative

```python
gamma_prime_pred
```

## Estimated de-warped wavelet transform

```python
x_wav_pred
```

---

# Evaluation Metrics

The repository computes:

* Mean Squared Error (MSE),
* Mean Absolute Error (MAE),
* stationarity criteria,


---

# Real-World Experiments

TWET was evaluated on:

* car acceleration recordings,
* singing voice,
* wind recordings.

Since no ground-truth deformation is available, evaluation relies on:

* stationarity restoration,
* qualitative analysis.

---

# Visualization

The repository provides visualizations for:

* signals,
* spectra,
* wavelet transforms,
* deformation functions,
* stationarity scores.

Example outputs include:

* estimated $\gamma'(t)$,
* reconstructed stationary signals,
* latent representations.

---

# GPU Support

TWET automatically uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

# Citation

If you use this repository, please cite:

```bibtex

```


---

# License



---

# Contact



