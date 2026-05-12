# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:59:26 2025

@author: coren
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


class BSM:
    def __init__(self, q=1.01, n_iter=20, grid_size=0, grid_width=5.0):
        """
        Parameters
        ----------
        q : float
            Base exponentielle (gamma' = q^delta)
        n_iter : int
            Nombre d’itérations
        grid_size : int
            Nombre de points dans la grille locale (impair recommandé)
        grid_width : float
            Largeur initiale de recherche
        """
        self.q = q
        self.n_iter = n_iter
        self.grid_size = grid_size
        self.grid_width = grid_width

    # --------------------------------------------------
    # INIT DELTA
    # --------------------------------------------------
    def estimate_delta_init(self, Y_wav):
        """
        Y_wav: (B, F, T)
        """
        B, F, T = Y_wav.shape
        m = np.arange(F)
    
        power = np.abs(Y_wav)**2  # (B,F,T)
    
        num = np.sum((self.q**m)[None,:,None] * power, axis=1)
        den = np.sum(power, axis=1) + 1e-3
    
        sigma = num / den  # (B,T)
    
        delta = np.log(sigma + 1e-3) / np.log(self.q)
        
        delta = (delta - np.mean(delta, axis=1, keepdims=True))
        
        return delta  

    # --------------------------------------------------
    # COVARIANCE
    # --------------------------------------------------
    def estimate_covariance(self, Y_wav):
        return np.mean(np.abs(Y_wav)**2, axis=2) + 1e-8  # (B,F)

    # --------------------------------------------------
    # SHIFT (vectorisé partiel)
    # --------------------------------------------------
    def shift_wavelet(self, W, delta):
        F = W.shape[0]
        m = np.arange(F)

        m_shift = m + delta

        m0 = np.floor(m_shift).astype(int)
        m1 = m0 + 1

        alpha = m_shift - m0

        m0 = np.clip(m0, 0, F - 1)
        m1 = np.clip(m1, 0, F - 1)

        return (1 - alpha) * W[m0] + alpha * W[m1]

    # --------------------------------------------------
    # ESTIMATION PRINCIPALE
    # --------------------------------------------------
    def estimate_delta(self, Y_wav):
        """
        Y_wav: (B,F,T)
        return delta: (B,T)
        """

        B, F, T = Y_wav.shape

        delta = self.estimate_delta_init(Y_wav)
        
        #"""
        grid_width = self.grid_width

        for k in range(self.n_iter):

            C = self.estimate_covariance(Y_wav)

            delta_new = np.zeros_like(delta)

            # grille locale
            half = self.grid_size // 2
            grid = np.linspace(-grid_width, grid_width, self.grid_size)

            for b in range(B):
                for t in range(T):

                    W = Y_wav[b, :, t]

                    best_score = np.inf
                    best_delta = delta[b, t]

                    for g in grid:
                        d = delta[b, t] + g

                        W_shift = self.shift_wavelet(W, d)

                        score = np.sum((np.abs(W_shift)**2) / C[b])

                        if score < best_score:
                            best_score = score
                            best_delta = d

                    delta_new[b, t] = best_delta

            # convergence
            err = np.linalg.norm(delta_new - delta) / (np.linalg.norm(delta) + 1e-12)
          
            

            if err < 1e-3:
                break

            # relaxation
            delta = 0.5 * delta_new + 0.5 * delta
            
            # annealing
            grid_width *= 0.5
            #"""
        return delta

    # --------------------------------------------------
    # RECONSTRUCTION
    # --------------------------------------------------
    def reconstruct_gamma(self, delta, Fs):
        """
        delta -> gamma, gamma'
        """
        gamma_prime = self.q ** delta

        gamma_prime /= np.mean(gamma_prime, axis=1, keepdims=True)

        B, T = gamma_prime.shape
        t = np.arange(T) / Fs

        gamma = np.zeros_like(gamma_prime)

        for b in range(B):
            gamma[b] = cumulative_trapezoid(gamma_prime[b], t, initial=0)

        return gamma, gamma_prime


    def zoh_upsample(self,x, factor, target_length=None):
        """
        x: (B, T_down)
        factor: facteur d'upsampling
        target_length: longueur finale souhaitée (optionnel)
    
        return: (B, T_up)
        """
        x_up = np.repeat(x, factor, axis=1)
    
        if target_length is not None:
            x_up = x_up[:, :target_length]
    
        return x_up
    # --------------------------------------------------
    # PIPELINE COMPLETE
    # --------------------------------------------------
    def fit(self, Y_wav,factor, Fs):
        """
        Y_wav: (B,F,T)
        """
        delta = self.estimate_delta(Y_wav)
        gamma, gamma_prime = self.reconstruct_gamma(delta, Fs)
        
        gamma = self.zoh_upsample(gamma, factor)
        gamma_prime = self.zoh_upsample(gamma_prime, factor)
        

        return gamma, gamma_prime, delta
  
    
  
    
  

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
 

    plt.rcdefaults()
    
    plt.rcParams.update({
        "font.size": 8,
        "legend.fontsize": 7,
        "axes.labelsize": 8,
    })
    
     
    
    
    
    import sys
    import os
    
    
    # Add project path for custom modules
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Common_Functions'))
    )
    
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..","JEFAS"))
    )

    
    from Time_Warping_Functions import TimeWarping1D,TimeWarping2D
    from Stationarity_Score import StationarityScore
    


    
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
 

    # =========================================================
    # LOAD DATASET
    # =========================================================
    
    gen = GenerateDatabase()
    real =True
    if real == False :
        file = "dataset.npz"
        nb_win=50
    else :
        file = "real_dataset.npz"
        nb_win=3
    file_path = os.path.join(db_path, file)
    
    data = gen.load_dataset(file_path)
    
    
    X = data["X_test"][:nb_win]
    Y = data["Y_test"][:nb_win]
    G = data["G_test"][:nb_win]
    G_prime = data["G_prime_test"][:nb_win]
    
    #nb_win = np.min([len(data["X_test"]),5])
    
    # =========================================================
    # ANALYSIS MODULES
    # =========================================================
    Fs = 44100
    if real == False:
        f_min, f_max = 20, Fs / 2
    else :
        f_min, f_max = 200, 3000
        
    N=X.shape[1]
    Ms = 50
    dec = 100
    T = int(N/dec) # nuombre d'échantillon apres décimation
    wavtyp = "sharp"# "sharp"|"dgauss"
    wavpar = 50

    wav = WaveletTransform(Fs, f_min, f_max, Ms, wavtyp)
    
    
    # =========================================================
    # WAVELET TRANSFORMS
    # =========================================================

    
    X_wav, t_axis_x, freqs_x = wav.cwt(X, wavpar,dec)
    Y_wav, t_axis_y, freqs_y = wav.cwt(Y, wavpar,dec)
    

    X_wav=np.abs(X_wav)
    Y_wav=np.abs(Y_wav)
    
 
    print("X_wav",X_wav.shape)
    print("X_wav",X_wav.shape) 
    print("X_wav",X_wav.shape) 
    
    

    #%%
    import time
    from JEFAS import JEFAS
    
    
    O = BSM(q=1.1, n_iter=15, grid_size=40, grid_width=10)#BSM(q=1.05, n_iter=15, grid_size=10, grid_width=2)
    J=JEFAS()


    G_prime_pred=[]
    X_wav_pred = []
    X_pred = []
    Score_X_JEFAS = []
    Score_Y_JEFAS = []
    
    TW2D = TimeWarping2D()
    TW1D = TimeWarping1D()
    # Estimation
    
    start = time.time()
    for id_sig in range(nb_win):
        
        _, g_prime_pred, _ = O.fit(
            Y_wav[id_sig:id_sig+1],dec,
            Fs=Fs
        )
        
        g_prime_pred=g_prime_pred.reshape(N)


        #g_prime_pred, _, _ = J.Jefas(Y[id_sig],dec,wavpar,freqmin=f_min/Fs,freqmax=f_max/Fs,Nf=Ms,wavtyp = wavtyp,Nit=10 )#,wavparamAM=wavparamAM)
        G_prime_pred.append(g_prime_pred)
        
    
        
        score_X_JEFAS=J.loss_JEFAS(Y[id_sig], g_prime_pred, wavparam=wavpar, freqmin=f_min/Fs, freqmax=f_max/Fs,Nf=Ms,Dt=dec)
        Score_X_JEFAS.append(score_X_JEFAS)
        
        score_Y_JEFAS=J.loss_JEFAS(Y[id_sig], np.ones(N), wavparam=wavpar, freqmin=f_min/Fs, freqmax=f_max/Fs,Nf=Ms,Dt=dec)
        Score_Y_JEFAS.append(score_Y_JEFAS)
        
        
        
        
        
    G_prime_pred=np.array(G_prime_pred)
    Score_X_JEFAS = np.array(Score_X_JEFAS)
    Score_Y_JEFAS = np.array(Score_Y_JEFAS)
    
    
    end = time.time()
    
    print(f"Elapsed time per signal: {(end - start)/(nb_win):.4f} s")
    
    X_wav_pred = TW2D.estimSpectrum_np(Y_wav, Fs, fmin=f_min, fmax=f_max, Ms=Ms,
                                               gamma_prime=G_prime_pred[:,::dec])
    
    
    X_pred = TW1D.stationarize_np(Y, G_prime_pred)
    
    #%%
    
    print("<MSE> (gamma'_pred-gamma')       ={:.8f}".format(np.mean(np.mean((G_prime_pred - G_prime)**2, axis=1))))
    print("<MAE> (gamma'_pred-gamma')       ={:.8f}".format(np.mean(np.abs(G_prime_pred - G_prime))))
    
    
    coefs_tronc = int(0.01*N)
    print(r"<MSE> (gamma'_pred-gamma') tronc={:.8f}".format(np.mean(np.mean((G_prime_pred[:,coefs_tronc:N-coefs_tronc] - G_prime[:,coefs_tronc:N-coefs_tronc] )**2, axis=1))))
    print(r"<MAE> (gamma'_pred-gamma') tronc={:.8f}".format(np.mean(np.abs(G_prime_pred[:,coefs_tronc:N-coefs_tronc] - G_prime[:,coefs_tronc:N-coefs_tronc]))))
    
    
    
  
    
    
    score = StationarityScore()
    
   

    Score_X_Y,Score_X,Score_Y,Stack=score.score_TWET_np(X_wav_pred,Y_wav)
    
    Score_Y = np.atleast_1d(Score_Y)
    Score_X = np.atleast_1d(Score_X)

    
    print("Score X pred TWET      = {:.5f}".format(np.mean(Score_X)))
    
    print("Score Y TWET           = {:.5f}".format(np.mean(Score_Y)))
    
    print("score X / score Y TWET = {:.5f}".format(np.mean(Score_X/Score_Y)))
    
   
        
    print("----")
    #print("Score_X_Y= {:.5f}".format(np.mean(Score_X_Y)))
            
      
    

    

    print("Score X pred JEFAS     = {:.8f}".format(np.mean(Score_X_JEFAS)))
    
    print("Score Y JEFAS          = {:.8f}".format(np.mean(Score_Y_JEFAS)))
    
    

    
    print("score Y / score X JEFAS= {:.5f}".format(np.mean(Score_Y_JEFAS/Score_X_JEFAS)))
    
    
    print("----")

    
    data_dict = {
        "X_pred": X_pred,
        "G_prime_pred": G_prime_pred
    }

    np.savez("data_BSM.npz", **data_dict)
   
    
   


    """
    score_X_TWET = [0.0980589 , 0.05168366, 0.04185242, 0.03803928, 0.07130882,
           0.06396071, 0.06749229, 0.04668629, 0.09712492, 0.03810999,
           0.09505204, 0.04343131, 0.10386302, 0.0427772 , 0.04181649,
           0.06485007, 0.04717007, 0.04207192, 0.07923952, 0.09936476,
           0.07406899, 0.03611609, 0.03493156, 0.07454633, 0.04018595,
           0.07167029, 0.12662972, 0.05695222, 0.05057699, 0.13589471,
           0.04229017, 0.08145605, 0.04077056, 0.04971826, 0.06765825,
           0.04969427, 0.08240922, 0.03781566, 0.08405586, 0.08607213,
           0.03700721, 0.03586719, 0.05248149, 0.07366969, 0.0552817 ,
           0.06715857, 0.05547621, 0.08677332, 0.08483903, 0.04926718]
    score_Y_TWET = [0.12797439, 0.06201358, 0.06015756, 0.06092735, 0.10176859,
           0.09299787, 0.10275121, 0.06542248, 0.13641807, 0.05482173,
           0.12501614, 0.04754708, 0.15309031, 0.05770527, 0.04931824,
           0.09423972, 0.06449787, 0.06339263, 0.11763285, 0.12630087,
           0.10486753, 0.04588149, 0.05317507, 0.10342386, 0.05536724,
           0.09871393, 0.18759768, 0.06974337, 0.07577483, 0.18460089,
           0.05301301, 0.13477177, 0.04478548, 0.05797048, 0.10252553,
           0.05722426, 0.13353761, 0.05563471, 0.13529442, 0.11439697,
           0.04341871, 0.04197844, 0.07854719, 0.11639419, 0.07335571,
           0.09703841, 0.07406769, 0.13236459, 0.11256338, 0.0695785 ]
    
    score_X_JEFAS = [-124.17034106, -294.65049293, -322.2810852 , -360.51203174,
           -228.76566928, -351.37997567, -300.27678028, -308.98344087,
           -257.92266831, -353.45253857, -327.56828426, -326.44192083,
           -247.92135299, -318.10837953, -321.41531794, -283.18077507,
           -302.40317166, -337.55599529, -229.25610089, -207.53229718,
           -263.9110336 , -338.83550658, -369.95996341, -307.99955268,
           -383.49163531, -249.65185634, -158.27629035, -206.91554614,
           -303.98161798, -204.08996049, -305.68114429, -265.83728285,
           -306.98714775, -290.99961255, -319.30705225, -280.95983039,
           -322.8385801 , -370.13291969, -255.66048602, -160.05902352,
           -309.79558394, -313.37536947, -348.78130306, -309.06758115,
           -306.27737259, -245.07153377, -282.1703154 , -318.66565852,
           -308.20875761, -283.19202713]
    score_Y_JEFAS = [-112.21008154, -287.78325107, -303.03881284, -332.75429405,
           -214.28946928, -328.55290332, -278.61096219, -291.21804095,
           -234.96263891, -338.9486498 , -310.64807207, -325.95267465,
           -227.2718893 , -300.21884161, -312.12527248, -264.85726746,
           -291.70929155, -316.00641088, -207.40863367, -202.39289535,
           -242.82244799, -330.30394286, -351.82651134, -289.28257087,
           -367.0350977 , -234.35467724, -123.04731854, -200.8986817 ,
           -274.18874162, -185.86475148, -301.68117671, -222.65156546,
           -301.27828602, -285.50606492, -301.83721287, -274.32099184,
           -287.71427984, -351.73632205, -220.46837963, -145.39626541,
           -308.51473768, -307.60683009, -324.92389602, -267.37002134,
           -299.50557601, -226.45971411, -273.57459318, -292.20337238,
           -287.86862281, -263.60773527]
    
    
    Elapsed time per signal: 3.1875 s
    <MSE> (gamma'_pred-gamma')       =0.10291569
    <MAE> (gamma'_pred-gamma')       =0.22341821
    <MSE> (gamma'_pred-gamma') tronc=0.10266359
    <MAE> (gamma'_pred-gamma') tronc=0.22324424
    Score X pred TWET      = 0.06391
    Score Y TWET           = 0.08943
    score X / score Y TWET = 0.72956
    ----
    Score X pred JEFAS     = -289.87920329
    Score Y JEFAS          = -272.49621476
    score Y / score X JEFAS= 0.93681
    
    """
            




  

     
    
    #%% test de reconstruction

    

    t=np.linspace(0,N-1,N)/Fs
    coeffs = 400
    
    coefs_tronc = int(0.01*N)
    for k in range(np.min([4,nb_win])) : 

       
        x=X[k]
        y=Y[k]
        
        x_pred = X_pred[k]
        
        score_x_pred = Score_X[k]
        score_y      = Score_Y[k]
        

        score_x_pred_JEFAS = Score_X_JEFAS[k]
        score_y_JEFAS = Score_Y_JEFAS[k]
        
        #print("Score_X",Score_X.shape)
        #print("Score_X_JEFAS",Score_X_JEFAS.shape)
        
        # =========================================================
        # SPECTRAL ANALYSIS
        # =========================================================
        fx, Sx = wav.spectrum(X[k])
        fy, Sy = wav.spectrum(Y[k])
        
        fx_pred, Sx_pred = wav.spectrum(X_pred[k])
        

        
        gamma_prime= G_prime[k]
        gamma_prime_pred= G_prime_pred[k]
        
        gamma= G[k]
        gamma_pred = cumulative_trapezoid(gamma_prime_pred, t, initial=0)
        
    
        # =========================================================
        # VISUALIZATION
        # =========================================================
        fig, axes = plt.subplots(5, 2, figsize=(12, 12))
    
        # -------------------------
        # Signals
        # -------------------------
        axes[0, 0].plot(t[:coeffs], x[:coeffs], label=r"$x(t)$")
        axes[0, 0].plot(t[:coeffs], x_pred[:coeffs], label=r"$\widetilde{x}(t)$")
        axes[0, 0].set_title(r"Signal $\widetilde{{x}}(t)$, score TWET = {:.4f}, JEFAS = {:4f}".format(score_x_pred,score_x_pred_JEFAS))
        axes[0, 0].set_xlabel("Time (s)")   
        axes[0, 0].set_ylabel("Amp")
        axes[0, 0].legend()
        axes[0, 0].grid()
    
        axes[0, 1].plot(t[:coeffs], y[:coeffs], label=r"$y(t)$")
        axes[0, 1].set_title(r"Signal $y(t)$, score = {:.4f}, JEFAS = {:4f}".format(score_y,score_y_JEFAS))
        axes[0, 1].set_xlabel("Time (s)")   
        axes[0, 1].set_ylabel("Amp")
        axes[0, 1].legend()
        axes[0, 1].grid()
    
        
        # -------------------------
        # error
        # -------------------------
        axes[1, 0].plot(t[:coeffs], x[:coeffs]-x_pred[:coeffs], label=r"$x(t)-\widetilde{x}(t$")
        axes[1, 0].set_title(r" $x(t)-\widetilde{{x}}(t)$, std = {:.4f}".format(np.std(x-x_pred)))
        axes[1, 0].set_xlabel("Time (s)")   
        axes[1, 0].set_ylabel("Amp")
        axes[1, 0].legend()
        axes[1, 0].grid()
    
        
    
        # -------------------------
        # Spectra
        # -------------------------
        axes[2, 0].plot(fx, Sx, label=r"$X(f)$")
        axes[2, 0].plot(fx_pred, Sx_pred, label=r"$\widetilde{X}(f)$")
        axes[2, 0].set_title("Spectrum X(f)")
        axes[2, 0].set_xlabel("Freq (Hz)")   
        axes[2, 0].set_ylabel("Amp")
        axes[2, 0].legend()
        axes[2, 0].grid()
    
        axes[2, 1].plot(fy, Sy, label="y")
        axes[2, 1].set_title("Spectrum Y(f)")
        axes[2, 1].set_xlabel("Freq (Hz)")   
        axes[2, 1].set_ylabel("Amp") 
        axes[2, 1].legend()
        axes[2, 1].grid()
    
        # -------------------------
        # Deformation
        # -------------------------
        axes[3, 0].plot(t[coefs_tronc:-coefs_tronc], gamma[coefs_tronc:-coefs_tronc], label=r"$\gamma(t)$")
        axes[3, 0].plot(t[coefs_tronc:-coefs_tronc], gamma_pred[coefs_tronc:-coefs_tronc], label=r"$\widetilde{\gamma}(t)$")
        axes[3, 0].set_title("Warping function")
        axes[3, 0].set_xlabel("Time (s)")   
        axes[3, 0].set_ylabel("Time (s)") 
        axes[3, 0].legend()
        axes[3, 0].grid()
    
        axes[3, 1].plot(t[coefs_tronc:-coefs_tronc], gamma_prime[coefs_tronc:-coefs_tronc], label=r"$\gamma'(t)$")
        axes[3, 1].plot(t[coefs_tronc:-coefs_tronc], gamma_prime_pred[coefs_tronc:-coefs_tronc], label=r"$\widetilde{\gamma}'(t)$")
        axes[3, 1].set_title(r"Warping derivative $\gamma'(t)-\widetilde{{\gamma}}'(t)$, std = {:.4f}".format(np.std(gamma_prime[coefs_tronc:-coefs_tronc]-gamma_prime_pred[coefs_tronc:-coefs_tronc])))
        axes[3, 1].set_xlabel("Time (s)")   
        axes[3, 1].set_ylabel("Freq (Hz)")
        axes[3, 1].legend()
        axes[3, 1].grid()
    
        # -------------------------
        # Wavelet transforms
        # -------------------------
       
        axes[4, 0].pcolormesh(t_axis_x,freqs_x, X_wav_pred[k], shading="auto")
        axes[4, 0].set_title(r"$W_x$")
        axes[4, 0].set_xlabel("Time (s)")   
        axes[4, 0].set_ylabel("Freq (Hz)") 
    
        axes[4, 1].pcolormesh(t_axis_x,freqs_x, Y_wav[k], shading="auto")
        axes[4, 1].set_title(r"$W_y$")
        axes[4, 1].set_xlabel("Time (s)")   
        axes[4, 1].set_ylabel("Freq (Hz)") 
    
        # =========================================================
        # DISPLAY
        # =========================================================
        plt.tight_layout()
        plt.show()
        
        
    
    
   
    
    
    