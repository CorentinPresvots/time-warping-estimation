# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:59:26 2025

@author: coren
"""
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid


class TimeWarping1D:
    def __init__(self, eps=1e-9):
        self.eps = eps

    # =========================================================
    # ---------------- TORCH VERSION --------------------------
    # =========================================================

    def cumtrapz(self, y, x):
        B, N = y.shape
        dx = x[1:] - x[:-1]
        avg = (y[:, 1:] + y[:, :-1]) / 2.0
        integral = torch.cumsum(avg * dx, dim=1)
        integral = torch.cat(
            [torch.zeros(B, 1, dtype=y.dtype, device=y.device), integral],
            dim=1
        )
        return integral

    def interp1d(self, x_coords, y_values, query):
        N = y_values.shape[1]

        idx = torch.searchsorted(x_coords, query, right=True)
        idx0 = torch.clamp(idx - 1, 0, N - 1)
        idx1 = torch.clamp(idx, 0, N - 1)

        y0 = torch.gather(y_values, 1, idx0)
        y1 = torch.gather(y_values, 1, idx1)
        t0 = torch.gather(x_coords, 1, idx0)
        t1 = torch.gather(x_coords, 1, idx1)

        denom = (t1 - t0).clamp(min=self.eps)
        w = (query - t0) / denom

        return (1.0 - w) * y0 + w * y1
    
 



    def stationarize_th(self, y, gamma_prime):
        """
        y: (B, N)
        gamma_prime: (B, N)
        returns x: (B, N)
        """
        B, N = y.shape
        dtype = y.dtype
        device = y.device
        tx = torch.linspace(0.0, 1.0, N, dtype=dtype, device=device)  # (N,)
        #print("tx", tx.shape)
        tx_batched = tx.unsqueeze(0).expand(B, -1)  # (B, N)
        #print("tx_batched",tx_batched.shape)
        # 1) gamma = integral gamma_prime
        
        #print("gamma_prime",gamma_prime.shape)
        gamma = self.cumtrapz(gamma_prime, tx)  # (B, N)
        #print("gamma",gamma.shape)
        
        # 2) inverse gamma: compute invgamma so that gamma(invgamma(t)) = t
        invgamma = self.interp1d(gamma, tx_batched, tx_batched) 
        #print("invgamma",invgamma.shape)
     
        # 3) invdgamma = interpolate gamma_prime at invgamma: invdgamma[b] = interp1d(tx, gamma_prime[b], invgamma[b])
        invdgamma = self.interp1d(tx_batched, gamma_prime, invgamma)

        # 4) y_warped = interpolate y at invgamma
        y_warped = self.interp1d(tx_batched, y, invgamma)

        x = y_warped / torch.sqrt(invdgamma + self.eps)
        return x
    
    
    
   
    
    def unstationarize_th(self, x, gamma_prime):
        """
        x: (B, N)
        gamma_prime: (B, N)
        returns y: (B, N)
        """
        B, N = x.shape
        device = x.device
        dtype = x.dtype
        tx = torch.linspace(0.0, 1.0, N, dtype=dtype, device=device)  # (N,)
        tx_batched = tx.unsqueeze(0).expand(B, -1)  # (B, N)
    
        gamma = self.cumtrapz_batch(gamma_prime, tx)  # (B, N)

        # x_warped(t) = x(gamma(t))
        x_warped = self.interp1d_batch(tx_batched, x, gamma)  # (B, N)

        safe_dgamma = torch.clamp(gamma_prime, min=self.eps)
        y = torch.sqrt(safe_dgamma) * x_warped
        return y






    # =========================================================
    # ---------------- NUMPY VERSION ---------------------------
    # =========================================================
    
    
    def _ensure_batch(self, x):
        """Ajoute une dimension batch si input (N,)"""
        if x.ndim == 1:
            return x[None, :], True
        return x, False
    

    def _restore_shape(self, x, was_1d):
        """Remove batch dimension if input was (N,)"""
        if was_1d:
            return x[0]
        return x

    def stationarize_np(self, y, gamma_prime):
        
        y, was_1d = self._ensure_batch(y)
        gamma_prime, _ = self._ensure_batch(gamma_prime)


        B, N = y.shape
        tx = np.linspace(0, 1, N)

        gamma = np.array([
            cumulative_trapezoid(gamma_prime[b], tx, initial=0)
            for b in range(B)
        ])

        invgamma = np.array([
            np.interp(tx, gamma[b], tx)
            for b in range(B)
        ])

        invdgamma = np.array([
            np.interp(invgamma[b], tx, gamma_prime[b])
            for b in range(B)
        ])

        y_warped = np.array([
            np.interp(invgamma[b], tx, y[b])
            for b in range(B)
        ])
        
        x= y_warped / np.sqrt(invdgamma + self.eps)
        return self._restore_shape(x, was_1d)

    def unstationarize_np(self, x, gamma_prime):
        
        x, was_1d = self._ensure_batch(x)
        gamma_prime, _ = self._ensure_batch(gamma_prime)

        B, N = x.shape
        tx = np.linspace(0, 1, N)

        gamma = np.array([
            cumulative_trapezoid(gamma_prime[b], tx, initial=0)
            for b in range(B)
        ])

        x_warped = np.array([
            np.interp(gamma[b], tx, x[b])
            for b in range(B)
        ])

        y = np.sqrt(np.maximum(gamma_prime, self.eps)) * x_warped

        return self._restore_shape(y, was_1d)





##################################################### sur le spectre :



class TimeWarping2D:
    def __init__(self, eps=1e-12):
        self.eps = eps

    # =========================================================
    # SHAPE HANDLING
    # =========================================================


    def _ensure_4d_np(self, X):
        """
        Ensure array is (B, C, F, T)
        """
        if X.ndim == 2:        # (F, T)
            return X[None, None, :, :], (True, True)
        elif X.ndim == 3:      # (B, F, T)
            return X[:, None, :, :], (False, True)
        elif X.ndim == 4:
            return X, (False, False)
        else:
            raise ValueError("Invalid array shape")

    def _restore_shape(self, X, flags):
        """
        Restore original shape
        """
        squeeze_batch, squeeze_channel = flags

        if squeeze_batch:
            X = X.squeeze(0)
        if squeeze_channel:
            X = X.squeeze(1)

        return X
    
        
        
    def _ensure_4d_wav(self, W):
        """
        Ensure W is (B,C,F,T)
    
        Accepts:
            (F,T)
            (B,F,T)
            (B,C,F,T)
        """
        squeeze_flags = {
            "batch": False,
            "channel": False
        }
    
        if W.ndim == 2:  # (F,T)
            W = W.unsqueeze(0).unsqueeze(0)
            squeeze_flags["batch"] = True
            squeeze_flags["channel"] = True
    
        elif W.ndim == 3:
            # ambiguity: (B,F,T) 
                # batch
                W = W.unsqueeze(1)
                squeeze_flags["channel"] = True
    
        elif W.ndim == 4:
            pass
    
        else:
            raise ValueError(f"Invalid Wy shape: {W.shape}")
    
        return W, squeeze_flags
    
    def _ensure_gamma(self, gamma_prime, B):
        """
        Ensure gamma_prime is (B,T)
        """
    
        if gamma_prime.ndim == 1:  # (T,)
            gamma_prime = gamma_prime.unsqueeze(0).expand(B, -1)
    
        elif gamma_prime.ndim == 2:  # (B,T)
            if gamma_prime.shape[0] != B:
                raise ValueError("Batch mismatch between Wy and gamma_prime")
    
        else:
            raise ValueError(f"Invalid gamma_prime shape: {gamma_prime.shape}")
    
        return gamma_prime
    
    def _restore_wav_shape(self, W, flags):
        if flags["channel"]:
            W = W.squeeze(1)
        if flags["batch"]:
            W = W.squeeze(0)
        return W

    # =========================================================
    # TORCH VERSION
    # =========================================================
    def estimSpectrum_th(self, Wy, Fs, fmin, fmax, Ms, gamma_prime):
    
        # -------------------------
        # Normalize inputs
        # -------------------------
        Wy, flags = self._ensure_4d_wav(Wy)
    
        B, C, F, T = Wy.shape
        
       
        gamma_prime = self._ensure_gamma(gamma_prime, B)  # (B,T)
    
        device = Wy.device
        dtype = Wy.dtype
    
        # -------------------------
        # Log-scales
        # -------------------------
        xi0 = Fs / 4.0
    
        smin = torch.log2(torch.tensor(xi0 / fmax, device=device, dtype=dtype))
        smax = torch.log2(torch.tensor(xi0 / fmin, device=device, dtype=dtype))
    
        scales_log = torch.linspace(smin, smax, Ms, device=device, dtype=dtype)  # (F,)
    
        # -------------------------
        # Time warping
        # -------------------------
        thetaTW = torch.log2(gamma_prime)  # (B,T)
    
        # (B,F,T)
        xq = scales_log[None, :, None] + thetaTW[:, None, :]
        xp = scales_log
    
        # -------------------------
        # Interpolation indices
        # -------------------------
        idx1 = torch.searchsorted(xp, xq)
        idx0 = (idx1 - 1).clamp(0, F - 1)
        idx1 = idx1.clamp(0, F - 1)
    
        # expand channels
        idx0 = idx0.unsqueeze(1).expand(-1, C, -1, -1)
        idx1 = idx1.unsqueeze(1).expand(-1, C, -1, -1)
    
        # -------------------------
        # Interpolation
        # -------------------------
        y0 = torch.gather(Wy, 2, idx0)
        y1 = torch.gather(Wy, 2, idx1)
    
        x0 = xp[idx0]
        x1 = xp[idx1]
    
        w = (xq.unsqueeze(1) - x0) / (x1 - x0 + self.eps)
    
        Wx = (1 - w) * y0 + w * y1
    
        # -------------------------
        # Mask invalid
        # -------------------------
        valid = (idx1 > 0) & (idx1 < F)
        Wx = Wx * valid
    
        # -------------------------
        # Restore original shape
        # -------------------------
        return self._restore_wav_shape(Wx, flags)

    # =========================================================
    # NUMPY VERSION
    # =========================================================
    def estimSpectrum_np(self, Wy, Fs, fmin, fmax, Ms, gamma_prime):

        Wy, flags = self._ensure_4d_np(Wy)

        if gamma_prime.ndim == 1:
            gamma_prime = gamma_prime[None, :]

        B, C, F, T = Wy.shape

        xi0 = Fs / 4
        smin = np.log2(xi0 / fmax)
        smax = np.log2(xi0 / fmin)

        scales_log = np.linspace(smin, smax, Ms)

        thetaTW = np.log2(gamma_prime)

        Wx = np.zeros_like(Wy)

        for b in range(B):
            for t in range(T):

                scalesX = scales_log + thetaTW[b, t]

                for c in range(C):

                    Wx[b, c, :, t] = np.interp(
                        scalesX,
                        scales_log,
                        Wy[b, c, :, t],
                        left=0,
                        right=0
                    )
        
        
       
        return self._restore_shape(Wx, flags)

    
if __name__ == "__main__":
    import sys
    import os

    # Add project path for custom modules
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Database'))
    )
    
    # path vers Database/
    db_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Database')
    )

    from Generate_Database import GenerateDatabase
    from CWT import WaveletTransform
    from Stationarity_Score import StationarityScore

    
    
    
    # =========================================================
    # LOAD DATASET
    # =========================================================
    
    gen = GenerateDatabase()

    file_path = os.path.join(db_path, "dataset.npz")
    
    
    data = gen.load_dataset(file_path)
    
    X = data["X_test"]
    Y = data["Y_test"]
    G = data["G_test"]
    G_prime = data["G_prime_test"]

    nb_win = np.min([5,len(X)])
    
    X=X[:nb_win]
    Y=Y[:nb_win]
    G=G[:nb_win]
    G_prime=G_prime[:nb_win]
    print("X shape", X.shape)
    
    
    
    # =========================================================
    # ANALYSIS MODULES
    # =========================================================
    Fs = 44100
    fmin, fmax = 20, Fs / 2
    Ms = 100
    wavtyp = "sharp"# "sharp"|"dgauss"
    wavpar = [5,10,50]

    wav = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
    
    
    # =========================================================
    # WAVELET TRANSFORMS
    # =========================================================
    W_x, t_axis_x, freqs_x = wav.cwt(X, wavpar)
    W_y, t_axis_y, freqs_y = wav.cwt(Y, wavpar)

    W_x_abs=np.abs(W_x)
    W_y_abs=np.abs(W_y)

    print("W_x",W_x_abs.shape)
 
    # =========================================================
    # Cumpute Scores np and torch
    # =========================================================
    StaScore = StationarityScore()
    sx_xy_mean_B,sx,sy,sx_xy = StaScore.score_TWET_np(W_x_abs,W_y_abs)
    
    
    #print("sx_xy_mean_B",sx_xy_mean_B.shape)
    #print("sx",sx.shape)
    #print("sx_xy",sx_xy.shape)


    print("mean_B stationary score x / y mean, numpy version",np.round(100000*sx_xy_mean_B)/100000)    
    #print("sx_xy",np.round(10000*sx_xy)/10000)  
    #print("sx",np.round(10000*sx)/10000) 
    #print("sy",np.round(10000*sy)/10000) 
    
    
    
    W_x_th = torch.from_numpy(W_x_abs).float()
    W_y_th = torch.from_numpy(W_y_abs).float()
    
    
    sx_xy_mean_B_th,sx_th,sy_th,sx_xy_th = StaScore.score_TWET_th(W_x_th,W_y_th)
    
    
    print("mean_B stationary score x / y mean, torch version",np.round(100000*sx_xy_mean_B_th.cpu().numpy())/100000)    
    #print("sx_xy_torch",np.round(10000*sx_xy.cpu().numpy())/10000)  
    
    
    
    #%% test time warping WY --> WX pred
    
    
    TW2D = TimeWarping2D()
    
    
    from CWT import WaveletTransform
    
    # =========================================================
    # ANALYSIS MODULES
    # =========================================================
    Fs = 44100
    fmin, fmax = 20, Fs / 2
    N=X.shape[1]
    Ms = 50
    dec = 100
    T = int(N/dec) # nuombre d'échantillon apres décimation
    wavtyp = "sharp"# "sharp"|"dgauss"
    wavpars = [20]

    wav = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
    
    
    # =========================================================
    # WAVELET TRANSFORMS
    # =========================================================
    X_wav, t_axis_x, freqs_x = wav.cwt(X, wavpars,dec)
    Y_wav, t_axis_y, freqs_y = wav.cwt(Y, wavpars,dec)
    
    X_wav = np.abs(X_wav)
    Y_wav = np.abs(Y_wav)
 
    print("X_wav",X_wav.shape)
    print("Y_wav",Y_wav.shape) 
 
    
    
    
    # ---------------------
    # 2. Conversion en tenseurs PyTorch
    # ---------------------
    import matplotlib.pyplot as plt
    import gc
             
    gc.collect()
    torch.cuda.empty_cache()  
    
    Y_th = torch.tensor(Y, dtype=torch.float32)
   
    
    Y_wav_th = torch.tensor(Y_wav, dtype=torch.float32)
    
    G_prime_th = torch.tensor(G_prime, dtype=torch.float32)
    
    
   
    batch_size=np.min([Y.shape[0],4])
    
    
    X_wav_pred_np = TW2D.estimSpectrum_np(Y_wav[:batch_size], Fs, fmin, fmax, Ms, G_prime[:batch_size,::dec])
    
    print("X_wav_pred_np.shape",X_wav_pred_np.shape)
    
  
    
    X_wav_pred_th = TW2D.estimSpectrum_th(Y_wav_th[:batch_size], 
                                                     Fs, fmin=fmin, fmax=fmax, Ms=Ms,
                             gamma_prime=G_prime_th[:batch_size,::dec])    
    
    
    
    
    print("X_wav_pred_th.shape",X_wav_pred_th.shape)
    
    X_wav_pred_th=X_wav_pred_th.cpu().numpy()
    
    channel_used = 0
    fig, axes = plt.subplots(batch_size,4, figsize=(18, 20))
    for k in range(batch_size):
         
        
        # ====== Spectrogramme ======
        ax_spec = axes[k,0]
        mesh = ax_spec.pcolormesh(t_axis_x, freqs_x,Y_wav[k,channel_used], shading='auto')
        ax_spec.set_ylabel("Fréquence (index)")
        ax_spec.set_xlabel("t (s)")
        ax_spec.set_title(f"CWT | wavpar {wavpars[channel_used]}")
        #fig.colorbar(mesh, ax=ax_spec)
        
        ax_spec = axes[k,1]
        mesh = ax_spec.pcolormesh(t_axis_x, freqs_x,X_wav_pred_np[k,channel_used], shading='auto')
        #ax_spec.set_ylabel("Fréquence (index)")
        ax_spec.set_xlabel("t (s)")
        ax_spec.set_title("CWT X np")
        #fig.colorbar(mesh, ax=ax_spec)       
        
        ax_spec = axes[k,2]
        mesh = ax_spec.pcolormesh(t_axis_x, freqs_x,X_wav_pred_th[k,channel_used], shading='auto')
        #ax_spec.set_ylabel("Fréquence (index)")
        ax_spec.set_xlabel("t (s)")
        ax_spec.set_title("CWT Y torch")
        #fig.colorbar(mesh, ax=ax_spec)
        
        ax_spec = axes[k,3]
        ax_spec.plot(np.linspace(0,N-1,N)/Fs,G_prime[k])
        ax_spec.set_xlabel("t (s)")
        ax_spec.set_title(r"$\gamma'(t)$")
        
        
        #print("energie Y {}".format(np.mean(Y_wav_train[kw]**2)))
        #print("energie X {}".format(np.mean(Sx[kw]**2)))
        
        
        
    

        
    
    #%%  test time warping 1D
 
    
   
    TW1D = TimeWarping1D()
    
    X_tilde_np=TW1D.stationarize_np(Y, G_prime)
    
    X_tilde_th=TW1D.stationarize_th(torch.tensor(Y), torch.tensor(G_prime))
    
    X_tilde_th=X_tilde_th.cpu().numpy()
        
    
     
    channel_used = 0
    fig, axes = plt.subplots(batch_size,2, figsize=(15, 20))
    coeffs= 300
    for k in range(batch_size):
        
      
        x_real = X[k]
        x_tilde_np=X_tilde_np[k]
        
        x_tilde_th=X_tilde_th[k]
        
       
        # ====== Spectrogramme ======
        ax_spec = axes[k,0]
        ax_spec.plot(x_real[:coeffs],label=r"$x(t)$")
        ax_spec.plot(x_tilde_np[:coeffs],label=r"$y(\gamma'^{-1}(t))$ np")
        ax_spec.plot(x_tilde_th[:coeffs],label=r"$y(\gamma'^{-1}(t))$ th")
        ax_spec.set_ylabel("Amp")
        ax_spec.set_xlabel("t (s)")
        ax_spec.set_title(r"Comparison between $x(t)$ and $y(\gamma'^{-1}(t))$")
        ax_spec.legend()
        
        ax_spec = axes[k,1]
        ax_spec.plot(x_real[:coeffs]-x_tilde_np[:coeffs],label=r"$x(t)-y(\gamma'^{-1}(t)))$ np")
        ax_spec.plot(x_real[:coeffs]-x_tilde_th[:coeffs],label=r"$x(t)-y(\gamma'^{-1}(t)))$ th")
        ax_spec.set_ylabel("Amp")
        ax_spec.set_xlabel("t (s)")
        ax_spec.set_title(r"Comparison between $x$ and $y(\gamma'^{-1})$")
        ax_spec.legend()
        
        
        
        
        
        
        
    
    