# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:41:53 2026

@author: coren
"""

import numpy as np
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class StationarityScore:
    def __init__(self, eps=1e-9):

        self.eps = eps

    # =======================
    #  VERSION TORCH
    # =======================
    
    def _ensure_batch_4d_th(self, X):
        """
        Ensure input is (B, C, F, T)
    
        Accepts:
            (F, T)       -> (1, 1, F, T)
            (C, F, T)    -> (1, C, F, T)
            (B, C, F, T) -> unchanged
        """
        if X.ndim == 2:        # (F, T)
            return X.unsqueeze(0).unsqueeze(0), True
        elif X.ndim == 3:      # (C, F, T)
            return X.unsqueeze(0), True
        elif X.ndim == 4:
            return X, False
        else:
            raise ValueError("Input must be 2D, 3D or 4D tensor")
        
        
    def score_TWET_th(self, Wx, Wy):
        """
        Compute TWET stationarity score (torch version)
    
        Accepts flexible input shapes:
            Wx, Wy:
                (F, T)
                (C, F, T)
                (B, C, F, T)
        """
    
        # --- Ensure proper shape
        Wx, squeeze = self._ensure_batch_4d_th(Wx)
        Wy, _       = self._ensure_batch_4d_th(Wy)
    
        # --- Shape
        B, C, F, T = Wx.shape
    
        # --- Time-frequency rearrangement
        psdx = Wx.permute(0, 1, 3, 2)  # (B,C,T,F)
        psdy = Wy.permute(0, 1, 3, 2)
    
        # --- Variability over time
        var_tx = psdx.std(dim=2)
        var_ty = psdy.std(dim=2)
    
        # --- Mean over frequency
        scoresx = var_tx.mean(dim=2, keepdim=True)
        scoresy = var_ty.mean(dim=2, keepdim=True)
    
        # --- Ratio
        scores = scoresx / (scoresy + self.eps)
    
        # --- Aggregate over channels
        mean_score = torch.mean(scores, dim=1)
    
        # --- Remove artificial batch dims if needed
        if squeeze:
            mean_score = mean_score.squeeze()
            scoresx = scoresx.squeeze()
            scoresy = scoresy.squeeze()
            scores  = scores.squeeze()
    
        return mean_score, scoresx, scoresy, scores

    # =======================
    #  VERSION NUMPY
    # =======================
    def _ensure_batch_4d(self, X):
        """
        Ensure input is (B, C, F, T)
        Accepts:
            (C, F, T) -> (1, C, F, T)
            (F, T)    -> (1, 1, F, T)
            (B, C, F, T) unchanged
        """
        if X.ndim == 2:      # (F, T)
            return X[None, None, :, :], True
        elif X.ndim == 3:    # (C, F, T)
            return X[None, :, :, :], True
        return X, False
    
    
    def score_TWET_np(self, Wx, Wy):
    
        Wx, squeeze = self._ensure_batch_4d(Wx)
        Wy, _ = self._ensure_batch_4d(Wy)
    
        # (B, C, T, F)
        psdx = np.transpose(Wx, (0, 1, 3, 2))
        psdy = np.transpose(Wy, (0, 1, 3, 2))
    
        # variance temporelle
        var_tx = np.std(psdx, axis=2)
        var_ty = np.std(psdy, axis=2)
    
        # moyenne sur fréquences
        scoresx = np.mean(var_tx, axis=2, keepdims=True)
        scoresy = np.mean(var_ty, axis=2, keepdims=True)
    
        scores = scoresx / (scoresy + self.eps)
    
        mean_score = np.mean(scores, axis=1)

        if squeeze:
            mean_score=mean_score.squeeze()
            scoresx  = scoresx.squeeze()
            scoresy = scoresy.squeeze()
            scores  = scores.squeeze()

        
        return mean_score, scoresx, scoresy, scores
    
    

    
    
    
    def _ensure_4d(self, W):
        """
        Ensure W is (B, C, F, T)
    
        Returns:
            W_4d
            flags = (squeeze_B, squeeze_C)
        """
        W = np.asarray(W)
    
        if W.ndim == 2:        # (F, T)
            return W[None, None, :, :], (True, True)
    
        elif W.ndim == 3:      # (B, F, T)
            return W[:, None, :, :], (False, True)
    
        elif W.ndim == 4:      # (B, C, F, T)
            return W, (False, False)
    
        else:
            raise ValueError(f"Invalid shape: {W.shape}")
        
    
        
    def _restore_score_shape(self, mean_score, scores, flags):
        """
        Restore shapes depending on original input
    
        mean_score:
            (B,C,1)
    
        scores:
            (B,C,T)
        """
    
        squeeze_B, squeeze_C = flags
    
        # --- reshape scores → add last dim (T,1)
        scores = scores[..., None]  # (B,C,T,1)
    
        if squeeze_B and squeeze_C:
            # (F,T) → scalar + (T,1)
            mean_score = mean_score.squeeze()
            scores = scores.squeeze(0).squeeze(0)
    
        elif squeeze_C:
            # (B,F,T) → (B,1) + (B,T,1)
            mean_score = mean_score.squeeze(1)  # (B,1)
            scores = scores.squeeze(1)          # (B,T,1)
    
        else:
            # (B,C,F,T) → (B,C,1) + (B,C,T,1)
            pass
    
        return mean_score, scores
    
    def score_JEFAS_simple(self, Wx, Wy, gamma_prime, wavtyp="sharp", wavparam=10):
        """
        Robust JEFAS stationarity score
    
        Accepts:
            Wx:
                (F,T)
                (B,F,T)
                (B,C,F,T)
    
            Wy:
                same shapes
    
            gamma_prime:
                (T,) or (B,T)
    
        Returns:
            mean_score
            score_t
        """
        
        #Wx=Wx.squeeze()
        #Wy=Wy.squeeze()
        # -----------------------------------
        # 1. Shape normalization
        # -----------------------------------
        Wx, flags = self._ensure_4d(Wx)
        Wy, _     = self._ensure_4d(Wy)
    
        B, C, F, T = Wx.shape
    
        gamma_prime = np.asarray(gamma_prime)
    
        if gamma_prime.ndim == 1:
            gamma_prime = gamma_prime[None, :]  # (1,T)
    
        if gamma_prime.shape[0] == 1:
            gamma_prime = np.repeat(gamma_prime, B, axis=0)
    
        assert gamma_prime.shape == (B, T)
    
        # -----------------------------------
        # 2. Spectrum estimation
        # -----------------------------------
        S = np.mean(np.abs(Wx)**2, axis=3)  # (B,C,F)
    
        # frequency grid
        omega = np.linspace(0, 2*np.pi, F)
    
        # -----------------------------------
        # 3. Wavelet covariance kernel
        # -----------------------------------
        scales = np.linspace(0.1, 1.0, F)
        M_tmp = np.outer(scales, omega)
    
        if wavtyp == "sharp":
            eps = np.finfo(float).eps
            Mpsi = np.exp(-2 * wavparam * (
                (np.pi / (2 * M_tmp + eps)) +
                (2 * M_tmp / np.pi) - 2
            ))
        elif wavtyp == "dgauss":
            Cst = 4 * wavparam / (np.pi**2)
            K = (2 / np.pi)**wavparam * np.exp(wavparam / 2)
            Mpsi = K * M_tmp**wavparam * np.exp(-Cst * M_tmp**2 / 2)
        else:
            raise ValueError("Unknown wavelet")
    
        Mpsi *= np.sqrt(scales[:, None])  # (F,F)
    
        # -----------------------------------
        # 4. Score computation
        # -----------------------------------
        scores = np.zeros((B, C, T))
    
        for b in range(B):
            for c in range(C):
    
                S_bc = S[b, c]  # (F,)
    
                for t in range(T):
    
                    delta = np.log2(gamma_prime[b, t])
    
                    # warped spectrum
                    Stheta = np.interp(
                        2**(-delta) * omega,
                        omega,
                        S_bc,
                        left=S_bc[0],
                        right=S_bc[-1]
                    )
    
                    # covariance
                    U = np.sqrt(Stheta)[None, :] * Mpsi  # (F,F)
                    Cmat = (U @ U.T.conj()) / F
    
                    # regularization
                    Cmat += 1e-8 * np.eye(F)
    
                    y = Wy[b, c, :, t:t+1]  # (F,1)
    
                    invC_y = np.linalg.solve(Cmat, y)
    
                    cost = (
                        np.linalg.slogdet(Cmat)[1]
                        + (y.T.conj() @ invC_y).real.item()
                    )
    
                    scores[b, c, t] = cost
    
        # -----------------------------------
        # 5. Aggregation
        # -----------------------------------
        mean_score = np.mean(scores, axis=2, keepdims=True)  # (B,C,1)
    
        # restore shape
        mean_score, scores = self._restore_score_shape(mean_score, scores, flags)
        return mean_score, scores
    
      
            
            
        
        
        
    
    
        
    
#%%

  
if __name__ == "__main__":
    import sys
    import os

    # Add project path for custom modules
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Database'))
    )
    
    from CWT import WaveletTransform
    import matplotlib.pyplot as plt
    from Generate_Signals import SignalGenerator
    

    # =========================================================
    # EXPERIMENT SETUP
    # =========================================================
    Fs = 44100
    T = 1
    N = int(Fs * T)

    f0_list = [Fs / 16, Fs / 8, Fs / 4]

    noise = False#True
    impulses = False#True

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
        gamma_kwargs["win_len"] = Fs // 50

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
    Ms = 50
    wavtyp = "dgauss"
    wavpar = 50
    dec = 10

    wav = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
    StaScore = StationarityScore()


    
    # =========================================================
    # WAVELET TRANSFORMS
    # =========================================================
    W_x, t_axis_x, freqs_x = wav.cwt(x, wavpar,dec)
    W_y, t_axis_y, freqs_y = wav.cwt(y, wavpar,dec)
    
    
    W_x_abs=np.abs(W_x)
    W_y_abs=np.abs(W_y)

    # =========================================================
    # Cumpute Scores np and torch
    # =========================================================
    _,score_x_np,score_y_np,_=StaScore.score_TWET_np(W_x_abs,W_y_abs)
    
    
    W_x_th = torch.from_numpy(W_x_abs).float()
    W_y_th = torch.from_numpy(W_y_abs).float()
    
    _,score_x_th,score_y_th,_=StaScore.score_TWET_th(W_x_th, W_y_th)#
    
    
    print("score X TWET = {:.8f}".format(score_x_np))
    print("score Y TWET = {:.8f}".format(score_y_np))
    #print("score_x_th = {:.8f}".format(score_x_th.cpu().numpy()))


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
    axes[0, 0].set_title(r"Signal $x(t)$, score = {:.4f}".format(score_x_np))
    axes[0, 0].set_xlabel("Time (s)")   
    axes[0, 0].set_ylabel("Amp")
    axes[0, 0].legend()
    axes[0, 0].grid()

    axes[0, 1].plot(t, y, label=r"$y(t)$")
    axes[0, 1].set_title(r"Signal $y(t)$, score = {:.4f}".format(score_y_np))
    axes[0, 1].set_xlabel("Time (s)")   
    axes[0, 1].set_ylabel("Amp")
    axes[0, 1].legend()
    axes[0, 1].grid()


    # -------------------------
    # Spectra
    # -------------------------
    axes[1, 0].plot(fx, Sx, label="x")
    axes[1, 0].set_title("Spectrum X(f)")
    axes[1, 0].set_xlabel("Freq (Hz)")   
    axes[1, 0].set_ylabel("Amp")
    axes[1, 0].legend()
    axes[1, 0].grid()

    axes[1, 1].plot(fy, Sy, label="y")
    axes[1, 1].set_title("Spectrum Y(f)")
    axes[1, 1].set_xlabel("Freq (Hz)")   
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
    axes[3, 0].pcolormesh(t_axis_x, freqs_x, np.abs(W_x.squeeze()), shading="auto")
    axes[3, 0].set_title(r"$W_x$")
    axes[3, 0].set_xlabel("Time (s)")   
    axes[3, 0].set_ylabel("Freq (Hz)") 

    axes[3, 1].pcolormesh(t_axis_x, freqs_x, np.abs(W_y.squeeze()), shading="auto")
    axes[3, 1].set_title(r"$W_y$")
    axes[3, 1].set_xlabel("Time (s)")   
    axes[3, 1].set_ylabel("Freq (Hz)") 

    # =========================================================
    # DISPLAY
    # =========================================================
    plt.tight_layout()
    plt.show()
    
    
    
    

    
  