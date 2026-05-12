# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:59:10 2025

@author: coren
"""

import numpy as np
import scipy.signal as sg
from scipy.integrate import cumulative_trapezoid
from numpy.polynomial import chebyshev as cheb

import sys
import os

# Add project path for custom modules
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Common_Functions'))
)

from Time_Warping_Functions import TimeWarping1D


class SignalGenerator:
    """
    Synthetic signal generator for time-warping analysis.

    This class generates:
    - stationary multiband signals in Fourier domain
    - smooth random deformation functions gamma'(t)
    - nonstationary signals via time warping
    """

    def __init__(self, Fs: float):
        self.Fs = Fs

    # =========================================================
    # GAMMA GENERATION
    # =========================================================
    def generate_gamma(self, N, std_gamma_prime=1/3, method="cos", seed=None, **kwargs):
        """
        Generate a smooth positive deformation derivative gamma'(t)
        and its integral gamma(t).
        """

        rng = np.random.default_rng(seed)
        t = np.linspace(0, N, N) / self.Fs

        # ---------------- Cosine mixture model ----------------
        if method == "cos":
            M = rng.integers(1, kwargs.get("M", 3) + 1)

            freqs = rng.uniform(1, 5, M)
            phases = rng.uniform(0, 2 * np.pi, M)
            amps = rng.uniform(0, 1, M)

            g = sum(
                a * np.cos(2 * np.pi * f * t + p)
                for f, p, a in zip(freqs, phases, amps)
            )

        # ---------------- Chebyshev polynomial model ----------------
        elif method == "chebyshev":
            deg = rng.integers(1, kwargs.get("deg", 5) + 1)
            coeffs = rng.uniform(-1, 1, deg + 1)

            g = cheb.chebval(2 * t * self.Fs / N - 1, coeffs)

        # ---------------- Filtered Gaussian noise ----------------
        elif method == "gauss":
            win_len = kwargs.get("win_len", self.Fs // 10)

            noise = rng.normal(0, 1, N)

            if win_len % 2 == 0:
                win_len += 1

            filt = sg.windows.hann(win_len)
            filt /= np.sum(filt)

            g = sg.convolve(noise, filt, mode="same")

        else:
            raise ValueError("Unknown gamma generation method")

        # ---------------- Normalization ----------------
        g = g - np.mean(g)
        g = g / (np.std(g) + 1e-12)

        # Ensure positivity of gamma'
        gamma_prime = np.abs(g * std_gamma_prime + 1)

        # Integrate to obtain gamma
        gamma = cumulative_trapezoid(gamma_prime, t, initial=0)

        return t, gamma, gamma_prime

    # =========================================================
    # STATIONARY SIGNAL GENERATION
    # =========================================================
    def generate_stationary(self, N, f0_list, margin_min=0, margin_max=10):
        """
        Generate stationary multiband signal in Fourier domain.
        """

        freqs = np.fft.fftfreq(N, 1 / self.Fs)
        Xf = np.zeros(N, dtype=np.complex128)

        for f0 in f0_list:

            width = np.random.uniform(margin_min, margin_max)

            mask = (freqs >= f0 - width / 2 - 1) & (freqs <= f0 + width / 2 + 1)
            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            amp = np.random.uniform(0.25, 1.0)

            hann = np.hanning(len(idx))
            phases = np.exp(1j * np.random.uniform(-np.pi, np.pi, len(idx)))

            Xf[idx] += amp * hann * phases

        # Enforce Hermitian symmetry to obtain real signal
        Xf = Xf + np.conj(np.roll(Xf[::-1], 1))

        x = np.fft.ifft(Xf).real
        x /= np.std(x) + 1e-12

        return x

    # =========================================================
    # OPTIONAL PERTURBATIONS
    # =========================================================
    def add_noise(self, x, sigma_range=(0.1, 0.5)):
        sigma = np.random.uniform(*sigma_range)
        return x + np.random.normal(0, sigma, size=len(x))

    def add_impulses(self, x, prob=1.0, K_range=(10, 40)):
        """
        Random short impulsive events (rare artifacts).
        """
        if np.random.rand() > prob:
            return x

        N = len(x)
        K = np.random.randint(*K_range)
        positions = np.random.randint(0, N, size=K)

        for pos in positions:
            duration = np.random.randint(1, 10)
            amplitude = np.random.uniform(1, 10)
            x[pos:pos + duration] += amplitude

        return x

    # =========================================================
    # FULL GENERATION PIPELINE
    # =========================================================
    def generate(
        self,
        N,
        f0_list,
        gamma_method="cos",
        std_gamma_prime=1/3,
        margin_min=0,
        margin_max=10,
        noise=False,
        impulses=False,
        seed=None,
        **kwargs
    ):
        """
        Full pipeline:
        1. Generate gamma / gamma'
        2. Generate stationary signal x
        3. Apply optional perturbations
        4. Apply time-warping to obtain y
        """

        # --- deformation model ---
        t, gamma, gamma_prime = self.generate_gamma(
            N,
            std_gamma_prime=std_gamma_prime,
            method=gamma_method,
            seed=seed,
            **kwargs
        )

        # --- stationary signal ---
        x = self.generate_stationary(N, f0_list, margin_min, margin_max)

        # --- optional corruption ---
        if noise:
            x = self.add_noise(x)

        if impulses:
            x = self.add_impulses(x)

        # --- time-warping (nonstationary signal) ---
        x_batch = np.array([x])
        gp_batch = np.array([gamma_prime])

        y = TimeWarping1D().unstationarize_np(x_batch, gp_batch)[0]

        return {
            "t": t,
            "x": x,
            "y": y,
            "gamma": gamma,
            "gamma_prime": gamma_prime
        }

#%%     
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from CWT import WaveletTransform
    from Stationarity_Score import StationarityScore
  

    # =========================================================
    # EXPERIMENT SETUP
    # =========================================================
    Fs = 44100
    T = 1
    N = int(Fs * T)

    f0_list = [Fs / 16, Fs / 8, Fs / 4]

    noise = True
    impulses = True

    margin_min = 0
    margin_max = 1000
    std_gamma_prime = 1 / 3

    gen = SignalGenerator(Fs)

    gamma_type = "gauss"  # "cos" | "chebyshev" | "gauss"

    # =========================================================
    # GENERATION OF DATA
    # =========================================================
    gamma_kwargs = {}

    if gamma_type == "cos":
        gamma_kwargs["M"] = 3

    elif gamma_type == "chebyshev":
        gamma_kwargs["deg"] = 5

    elif gamma_type == "gauss":
        gamma_kwargs["win_len"] = Fs // 5

    data = gen.generate(
        N=N,
        f0_list=f0_list,
        gamma_method=gamma_type,
        std_gamma_prime=std_gamma_prime,
        margin_min=margin_min,
        margin_max=margin_max,
        noise=noise,
        impulses=impulses,
        **gamma_kwargs
    )

    # =========================================================
    # UNPACK DATA
    # =========================================================
    t = data["t"]
    x = data["x"]
    y = data["y"]
    gamma = data["gamma"]
    gamma_prime = data["gamma_prime"]

    # =========================================================
    # ANALYSIS MODULES
    # =========================================================
    fmin, fmax = 20, Fs / 2
    Ms = 500
    wavtyp = "dgauss"
    wavpar = 50

    wav = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
    StaScore = StationarityScore()
    TW1D = TimeWarping1D()

    # =========================================================
    # WAVELET TRANSFORMS
    # =========================================================
    W_x, t_axis_x, freqs_x = wav.cwt(x, wavpar)
    W_y, t_axis_y, freqs_y = wav.cwt(y, wavpar)

    # =========================================================
    # TIME WARPING VALIDATION
    # =========================================================
    y_tilde = TW1D.unstationarize_np(x, gamma_prime)
    x_tilde = TW1D.stationarize_np(y, gamma_prime)

    # =========================================================
    # Stationarity scores
    # =========================================================
    
    _,score_x,score_y,_ = StaScore.score_TWET_np(np.abs(W_x),np.abs(W_y))
   
        
    # =========================================================
    # SPECTRAL ANALYSIS
    # =========================================================
    fx, Sx = wav.spectrum(x)
    fy, Sy = wav.spectrum(y)

    fx, Sx_tilde = wav.spectrum(x_tilde)
    fy, Sy_tilde = wav.spectrum(y_tilde)

    # =========================================================
    # VISUALIZATION
    # =========================================================
    fig, axes = plt.subplots(5, 2, figsize=(12, 12))

    # -------------------------
    # Signals
    # -------------------------
    axes[0, 0].plot(t, x, label=r"$x(t)$")
    axes[0, 0].plot(t, x_tilde, label=r"$\widetilde{x}(t)=y(\gamma'^{-1}(t))$")
    axes[0, 0].set_title(r"Signal $x(t)$, score = {:.4f}".format(score_x))
    axes[0, 0].set_xlabel("Time (s)")   
    axes[0, 0].set_ylabel("Amp") 
    axes[0, 0].legend()
    axes[0, 0].grid()

    axes[0, 1].plot(t, y, label=r"$y(t)$")
    axes[0, 1].plot(t, y_tilde, label=r"$\widetilde{y}(t)=y(\gamma'(t))$")
    axes[0, 1].set_title(r"Signal $y(t)$, score = {:.4f}".format(score_y))
    axes[0, 1].set_xlabel("Time (s)")   
    axes[0, 1].set_ylabel("Amp") 
    axes[0, 1].legend()
    axes[0, 1].grid()

    # -------------------------
    # Reconstruction error
    # -------------------------
    axes[1, 0].plot(t, x - x_tilde, label=r"$x-\tilde{x}$")
    axes[1, 0].set_title("Reconstruction error (x)")
    axes[1, 0].set_xlabel("Time (s)")   
    axes[1, 0].set_ylabel("Amp") 
    axes[1, 0].legend()
    axes[1, 0].grid()

    axes[1, 1].plot(t, y - y_tilde, label=r"$y-\tilde{y}$")
    axes[1, 1].set_title("Reconstruction error (y)")
    axes[1, 1].set_xlabel("Time (s)")   
    axes[1, 1].set_ylabel("Amp") 
    axes[1, 1].legend()
    axes[1, 1].grid()

    # -------------------------
    # Spectra
    # -------------------------
    axes[2, 0].plot(fx, Sx, label="x")
    axes[2, 0].plot(fx, Sx_tilde, label=r"$\tilde{x}$")
    axes[2, 0].set_title("Spectrum X(f)")
    axes[2, 0].set_xlabel("Time (s)")   
    axes[2, 0].set_ylabel("Amp") 
    axes[2, 0].legend()
    axes[2, 0].grid()

    axes[2, 1].plot(fy, Sy, label="y")
    axes[2, 1].plot(fy, Sy_tilde, label=r"$\tilde{y}$")
    axes[2, 1].set_title("Spectrum Y(f)")
    axes[2, 1].set_xlabel("Time (s)")   
    axes[2, 1].set_ylabel("Amp") 
    axes[2, 1].legend()
    axes[2, 1].grid()

    # -------------------------
    # Deformation
    # -------------------------
    axes[3, 0].plot(t, gamma, label=r"$\gamma(t)$")
    axes[3, 0].set_title("Warping function")
    axes[3, 0].set_xlabel("Time (s)")   
    axes[3, 0].set_ylabel("Time (s)") 
    axes[3, 0].legend()
    axes[3, 0].grid()

    axes[3, 1].plot(t, gamma_prime, label=r"$\gamma'(t)$")
    axes[3, 1].set_title("Warping derivative")
    axes[3, 1].set_xlabel("Time (s)")   
    axes[3, 1].set_ylabel("Freq (Hz)") 
    axes[3, 1].legend()
    axes[3, 1].grid()

    # -------------------------
    # Wavelet transforms
    # -------------------------
    dt = t_axis_x[1] - t_axis_x[0]

    t_edges = np.concatenate([t_axis_x - dt / 2, [t_axis_x[-1] + dt / 2]])

    y_edges = np.concatenate([
        [freqs_x[0] - (freqs_x[1] - freqs_x[0]) / 2],
        0.5 * (freqs_x[:-1] + freqs_x[1:]),
        [freqs_x[-1] + (freqs_x[-1] - freqs_x[-2]) / 2],
    ])

    axes[4, 0].pcolormesh(t_edges, y_edges, np.abs(W_x), shading="auto")
    axes[4, 0].set_title(r"$W_x$")
    axes[4, 0].set_xlabel("Time (s)")   
    axes[4, 0].set_ylabel("Freq (Hz)") 
    

    axes[4, 1].pcolormesh(t_edges, y_edges, np.abs(W_y), shading="auto")
    axes[4, 1].set_title(r"$W_y$")
    axes[4, 1].set_xlabel("Time (s)")   
    axes[4, 1].set_ylabel("Freq (Hz)")

    # =========================================================
    # DISPLAY
    # =========================================================
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    