# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:40:06 2025

@author: coren
"""

import numpy as np
from scipy.special import gamma


class WaveletTransform:
    def __init__(self, Fs, fmin, fmax, Ms, wavtyp="dgauss"):
        """
        Parameters
        ----------
        Fs : float
            Sampling frequency
        fmin, fmax : float
            Frequency range
        Ms : int
            Number of scales
        wavtyp : str
            'sharp' or 'dgauss'
        """
        self.Fs = Fs
        self.fmin = fmin
        self.fmax = fmax
        self.Ms = Ms
        self.wavtyp = wavtyp
      

        self._compute_scales()

    # =========================================================
    # Scales & frequencies
    # =========================================================
    def _compute_scales(self):
        xi0 = self.Fs / 4.0

        smin = np.log2(xi0 / self.fmax)
        smax = np.log2(xi0 / self.fmin)

        self.scales = 2.0 ** np.linspace(smin, smax, self.Ms)
        self.freqs = xi0 / self.scales

        # ensure increasing frequencies
        if self.freqs[0] > self.freqs[-1]:
            self.freqs = self.freqs[::-1]
            self.scales = self.scales[::-1]

    # =========================================================
    # Wavelet in Fourier domain
    # =========================================================
    def _wavelet_ft(self, scales, fff,wavpar):
        tmp = np.outer(scales, fff)

        if self.wavtyp == "sharp":
            eps = np.finfo(np.float64).eps
            fpsi = np.exp(-2 * wavpar * (
                (np.pi / (2 * tmp + eps)) + (2 * tmp / np.pi) - 2
            ))

        elif self.wavtyp == "dgauss":
            Cst = 4 * wavpar / (np.pi**2)
            K = (2 / np.pi)**wavpar * np.exp(wavpar / 2)
            fpsi = K * tmp**wavpar * np.exp(-Cst * tmp**2 / 2)

        else:
            raise ValueError("Unknown wavelet type")

        return fpsi

    # =========================================================
    # CWT
    # =========================================================
    def _ensure_signal_batch(self, sig):
        """Ensure (B, N)"""
        sig = np.asarray(sig)
        if sig.ndim == 1:
            return sig[None, :], True
        return sig, False
    
    
    def _ensure_wavelet_params(self, wavpars):
        """Ensure (C,)"""
        wavpars = np.asarray(wavpars)
        if wavpars.ndim == 0:
            return wavpars[None], True
        return wavpars, False
    
    
    def _restore_cwt_shape(self, W, was_1d, was_scalar):
        """
        Restore output shape:
        (B,C,F,T) → depending on inputs
        """
        if was_1d and was_scalar:
            return W[0, 0]          # (F,T)
        elif was_1d:
            return W[0]             # (C,F,T)
        elif was_scalar:
            return W[:, 0]          # (B,F,T)
        return W                   # (B,C,F,T)
    
    
    def cwt(self, sigs, wavpars, dec=100):

        sigs, was_1d = self._ensure_signal_batch(sigs)
        wavpars, was_scalar = self._ensure_wavelet_params(wavpars)
    
        B, N = sigs.shape
        C = len(wavpars)
        F = self.Ms
    
        t = np.arange(N) / self.Fs
        t_dec = t[::dec]
    
        fff = np.arange(N) * 2 * np.pi / N
        fsig = np.fft.fft(sigs, axis=1)  # (B,N)
    
        scales = self.scales.reshape(F, 1)
    
        # full resolution first
        W_full = np.zeros((B, C, F, N), dtype=np.complex128)
    
        for c, wavpar in enumerate(wavpars):
    
            fpsi = self._wavelet_ft(self.scales, fff, wavpar)  # (F,N)
            U = np.sqrt(scales) * fpsi                         # (F,N)
    
            for b in range(B):
                fTrans = U * fsig[b][None, :]
                W_full[b, c] = np.fft.ifft(fTrans)
    
        # --- decimation
        W = W_full[..., ::dec]  # (B,C,F,N//dec)
    
        return self._restore_cwt_shape(W, was_1d, was_scalar), t_dec, self.freqs
    # =========================================================
    # ICWT
    # =========================================================
    def _ensure_cwt_4d(self, W):
        """
        Ensure (B,C,F,T)
        """
        W = np.asarray(W)
    
        if W.ndim == 2:      # (F,T)
            return W[None, None, :, :], (True, True)
        elif W.ndim == 3:    # (C,F,T)
            return W[None, :, :, :], (True, False)
        elif W.ndim == 4:
            return W, (False, False)
        else:
            raise ValueError("Invalid shape")
    
    
    def _restore_icwt_shape(self, y, flags):
        squeeze_batch, squeeze_channel = flags
    
        if squeeze_batch:
            y = y.squeeze(0)
        if squeeze_channel:
            y = y.squeeze(0)
    
        return y
    
    
    # =========================================================
    # GENERAL ICWT
    # =========================================================
    def icwt(self, W, wavpars, dec=100):
    
        W, flags = self._ensure_cwt_4d(W)
        wavpars, was_scalar = self._ensure_wavelet_params(wavpars)
    
        B, C, F, T_dec = W.shape
        T = T_dec * dec
    
        # --- interpolation temporelle
        t_dec = np.arange(T_dec)
        t_full = np.linspace(0, T_dec - 1, T)
    
        W_full = np.zeros((B, C, F, T), dtype=W.dtype)
    
        for b in range(B):
            for c in range(C):
                for f in range(F):
                    W_full[b, c, f] = np.interp(
                        t_full,
                        t_dec,
                        W[b, c, f].real
                    ) + 1j * np.interp(
                        t_full,
                        t_dec,
                        W[b, c, f].imag
                    )
    
        # --- reconstruction classique
        fff = np.arange(T) * 2 * np.pi / T
        scales = self.scales.reshape(F, 1)
    
        ds = np.log(scales[1]) - np.log(scales[0])
        tmp = np.outer(scales, fff)
    
        y = np.zeros((B, C, T))
    
        for c, wavpar in enumerate(wavpars):
    
            if self.wavtyp == "sharp":
                eps = np.finfo(np.float64).eps
                fpsi = np.exp(-2 * wavpar * (
                    (np.pi / (2 * tmp + eps)) + (2 * tmp / np.pi) - 2
                ))
                Cpsi = 0.88 / np.sqrt(wavpar)
    
            elif self.wavtyp == "dgauss":
                Cst = 4 * wavpar / (np.pi**2)
                K = (2 / np.pi)**wavpar * np.exp(wavpar / 2)
                fpsi = K * tmp**wavpar * np.exp(-Cst * tmp**2 / 2)
                Cpsi = (K**2 / (2 * Cst**wavpar)) * gamma(wavpar)
    
            else:
                raise ValueError("Unknown wavelet type")
    
            for b in range(B):
    
                fy = fpsi * np.fft.fft(W_full[b, c], axis=1)
                fy = 2 * np.fft.ifft(fy).real
                fy = fy / np.sqrt(scales)
    
                y[b, c] = (1 / Cpsi) * np.sum(fy, axis=0) * ds
    
        if was_scalar:
            y = y[:, 0]
    
        return self._restore_icwt_shape(y, flags)

    # =========================================================
    # Power spectrum
    # =========================================================
    def spectrum(self, y):
        y = np.asarray(y)
        N = len(y)
    
        Y = np.fft.fft(y)
        f = np.fft.fftfreq(N, d=1 / self.Fs)
    
        idx = np.argsort(f)
    
        S = 2 * (np.abs(Y[idx]) ** 2) / (N ** 2)
        
    
        return f[idx][N//2:], S[N//2:]
    
#%%
   
    
if __name__ == "__main__":
    import sys
    import os

    # Add project path for custom modules
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Database'))
    )
    
    
    import matplotlib.pyplot as plt
    from Generate_Signals import SignalGenerator
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
        gamma_kwargs["win_len"] = Fs // 10

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
    Ms = 100
    dec = 100
    wavtyp = "dgauss"
    wavpar = 50

    wav = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
    StaScore = StationarityScore()

    # =========================================================
    # WAVELET TRANSFORMS
    # =========================================================
    W_x, t_axis_x, freqs_x = wav.cwt(x, wavpar,dec)
    W_y, t_axis_y, freqs_y = wav.cwt(y, wavpar,dec)
    
    print("W_x",W_x.shape)
    W_x_abs=np.abs(W_x)
    W_y_abs=np.abs(W_y)

    # =========================================================
    # Cumpute Scores np and torch
    # =========================================================
    _,score_x,score_y,_=StaScore.score_TWET_np(W_x_abs,W_y_abs)

    # =========================================================
    # SPECTRAL ANALYSIS
    # =========================================================
    fx, Sx = wav.spectrum(x)
    fy, Sy = wav.spectrum(y)

    # =========================================================
    # VISUALIZATION
    # =========================================================
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))

    # -------------------------
    # Signals
    # -------------------------
    axes[0, 0].plot(t, x, label=r"$x(t)$")
    axes[0, 0].set_title(r"Signal $x(t)$, score = {:.4f}".format(score_x))
    axes[0, 0].set_xlabel("Time (s)")   
    axes[0, 0].set_ylabel("Amp") 
    axes[0, 0].legend()
    axes[0, 0].grid()

    axes[0, 1].plot(t, y, label=r"$y(t)$")
    axes[0, 1].set_title(r"Signal $y(t)$, score = {:.4f}".format(score_y))
    axes[0, 1].set_xlabel("Time (s)")   
    axes[0, 1].set_ylabel("Amp") 
    axes[0, 1].legend()
    axes[0, 1].grid()


    # -------------------------
    # Spectra
    # -------------------------
    axes[1, 0].plot(fx, Sx, label="x")
    axes[1, 0].set_title("Spectrum X(f)")
    axes[1, 0].set_xlabel("Time (s)")   
    axes[1, 0].set_ylabel("Amp") 
    axes[1, 0].legend()
    axes[1, 0].grid()

    axes[1, 1].plot(fy, Sy, label="y")
    axes[1, 1].set_title("Spectrum Y(f)")
    axes[1, 1].set_xlabel("Time (s)")   
    axes[1, 1].set_ylabel("Amp") 
    axes[1, 1].legend()
    axes[1, 1].grid()

    # -------------------------
    # Deformation
    # -------------------------
    axes[2, 0].plot(t, gamma, label=r"$\gamma(t)$")
    axes[2, 0].set_title("Warping function")
    axes[2, 0].set_xlabel("Time (s)")   
    axes[2, 0].set_ylabel("Time (s)") 
    axes[2, 0].legend()
    axes[2, 0].grid()

    axes[2, 1].plot(t, gamma_prime, label=r"$\gamma'(t)$")
    axes[2, 1].set_title("Warping derivative")
    axes[2, 1].set_xlabel("Time (s)")   
    axes[2, 1].set_ylabel("Freq (Hz)") 
    axes[2, 1].legend()
    axes[2, 1].grid()

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

    axes[3, 0].pcolormesh(t_edges, y_edges, np.abs(W_x), shading="auto")
    axes[3, 0].set_title(r"$W_x$")
    axes[3, 0].set_xlabel("Time (s)")   
    axes[3, 0].set_ylabel("Freq (Hz)") 

    axes[3, 1].pcolormesh(t_edges, y_edges, np.abs(W_y), shading="auto")
    axes[3, 1].set_title(r"$W_y$")
    axes[3, 1].set_xlabel("Time (s)")   
    axes[3, 1].set_ylabel("Freq (Hz)") 

    # =========================================================
    # DISPLAY
    # =========================================================
    plt.tight_layout()
    plt.show()
    
    