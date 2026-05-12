# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:59:26 2025

@author: coren
"""
import numpy as np
from Generate_Signals import SignalGenerator


class GenerateDatabase(SignalGenerator):
    """
    Class for generating synthetic datasets of stationary / non-stationary signals
    using time-warping transformations.

    Inherits from SignalGenerator to reuse:
        - gamma generation
        - stationary signal generation
        - time-warping (x -> y)
    """

    def __init__(self, Fs=44100):
        """
        Parameters
        ----------
        Fs : float
            Sampling frequency
        """
        super().__init__(Fs)

    # =========================================================
    # FREQUENCY SAMPLING
    # =========================================================
    def generate_f0(self, nb_freq, f_min=50, f_max=100, margin=20):
        """
        Generate a list of central frequencies with minimum spacing.

        The method ensures that frequencies are separated by at least 'margin'.

        Parameters
        ----------
        nb_freq : int
            Maximum number of frequencies
        f_min, f_max : float
            Frequency range
        margin : float
            Minimum spacing between frequencies

        Returns
        -------
        f0_list : ndarray
            Sorted array of selected frequencies
        """

        available_intervals = [(f_min, f_max)]
        f0_list = []

        for _ in range(nb_freq):

            # Keep only intervals large enough
            valid_intervals = [
                (a, b) for (a, b) in available_intervals if (b - a) > margin
            ]

            if len(valid_intervals) == 0:
                break

            # Randomly pick an interval
            a, b = valid_intervals[np.random.randint(len(valid_intervals))]

            # Sample frequency inside interval
            f0 = np.random.uniform(a, b)
            f0_list.append(f0)

            # Update available intervals (remove neighborhood of f0)
            new_intervals = []
            for (x1, x2) in available_intervals:

                if (x1, x2) == (a, b):

                    if x1 < f0 - margin:
                        new_intervals.append((x1, f0 - margin / 2))

                    if f0 + margin < x2:
                        new_intervals.append((f0 + margin / 2, x2))
                else:
                    new_intervals.append((x1, x2))

            available_intervals = new_intervals

        return np.sort(np.array(f0_list))

    # =========================================================
    # DATASET GENERATION
    # =========================================================
    def generate_dataset(
        self,
        nb_win,
        T,
        Fs,
        type_gamma,
        noise=False,
        impulses=False,
    ):
        """
        Generate a dataset of synthetic signals.

        Each sample consists of:
            - x : stationary signal
            - y : time-warped signal
            - gamma : warping function
            - gamma_prime : warping derivative

        Parameters
        ----------
        nb_win : int
            Number of samples to generate
        T : float
            Signal duration (seconds)
        Fs : float
            Sampling frequency
        type_gamma : list
            List of gamma generation methods ("cos", "chebyshev", "gauss")
        noise : bool
            Add Gaussian noise
        impulses : bool
            Add impulsive noise

        Returns
        -------
        X, Y, G, G_prime : ndarray
            Arrays of shape (nb_win, N)
        """

        N = int(np.ceil(Fs * T))

        # Frequency range for signal generation
        f_min = Fs / 300
        f_max = Fs / 4

        max_nb_freqs = 5
        margin_min = 1
        margin_max = Fs / 40

        X, Y, G, G_prime = [], [], [], []

        nb_w = 0
        while nb_w < nb_win:

            # Random number of frequency components
            nb_freqs = np.random.randint(1, max_nb_freqs+1)

            # Warping variability
            std_gamma_prime = 1 / 3

            # Generate frequency components
            f0_list = self.generate_f0(
                nb_freqs,
                f_min=f_min,
                f_max=f_max,
                margin=margin_max,
            )

            # Random gamma type
            gamma_type = np.random.choice(type_gamma)

            gamma_kwargs = {}

            if gamma_type == "cos":
                gamma_kwargs["M"] = np.random.randint(1, 4)

            elif gamma_type == "chebyshev":
                gamma_kwargs["deg"] = np.random.randint(1, 6)

            elif gamma_type == "gauss":
                gamma_kwargs["win_len"] = int(Fs/np.random.uniform(1, 60))

            # Generate one sample
            data = self.generate(
                N=N,
                f0_list=f0_list,
                gamma_method=gamma_type,
                std_gamma_prime=std_gamma_prime,
                margin_min=margin_min,
                margin_max=margin_max,
                noise=noise,
                impulses=impulses,
                **gamma_kwargs,
            )

            # Store results
            X.append(data["x"])
            Y.append(data["y"])
            G.append(data["gamma"])
            G_prime.append(data["gamma_prime"])

            nb_w += 1

        # Convert to arrays
        X = np.array(X)
        Y = np.array(Y)
        G = np.array(G)
        G_prime = np.array(G_prime)
        
        
        return {
            "X": X,
            "Y": Y,
            "G": G,
            "G_prime": G_prime
        }
        
    
    def save_dataset(self, filepath, data_dict):
        """
        Save dataset to a compressed .npz file.
    
        Parameters
        ----------
        filepath : str
            Path to save the dataset
        data_dict : dict
            Dictionary containing dataset arrays
        """
        np.savez_compressed(filepath, **data_dict)
    
    def load_dataset(self, filepath):
        """
        Load dataset from a .npz file.
    
        Parameters
        ----------
        filepath : str
    
        Returns
        -------
        data_dict : dict
        """
        
        
        data = np.load(filepath)
    
        return {key: data[key] for key in data.files}
    
      
    def generate_and_save_dataset(self, filepath,
                               nb_train, nb_val, nb_test,
                               T, Fs, type_gamma,
                               noise=False, impulses=False):
        """
        Generate full dataset (train/val/test) and save it.
        """
    
        data_train = self.generate_dataset(
            nb_train, T, Fs, type_gamma, noise, impulses
        )
    
        data_val = self.generate_dataset(
            nb_val, T, Fs, type_gamma, noise, impulses
        )
    
        data_test = self.generate_dataset(
            nb_test, T, Fs, type_gamma, noise, impulses
        )
    
        data_dict = {
            "X_train": data_train["X"],
            "Y_train": data_train["Y"],
            "G_train": data_train["G"],
            "G_prime_train": data_train["G_prime"],
            "X_val": data_val["X"],
            "Y_val": data_val["Y"],
            "G_val": data_val["G"],
            "G_prime_val": data_val["G_prime"],
            "X_test": data_test["X"],
            "Y_test": data_test["Y"],
            "G_test": data_test["G"],
            "G_prime_test": data_test["G_prime"]
        }
    
        self.save_dataset(filepath, data_dict)
    
        return data_dict
    
    def save_real_dataset(self,folder,Fs,N_target,filepath):
        import os
        from scipy.io import loadmat
        

        
        # -------------------------------------------------
        # LOAD + CLEAN SIGNALS
        # -------------------------------------------------
        Y_list = []
        
        files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])
        
        for file in files:
            path = os.path.join(folder, file)
            mat = loadmat(path)
        
            if "y" not in mat:
                continue
        
            y = np.asarray(mat["y"])
            y = np.squeeze(y).astype(np.float64)
            y = np.nan_to_num(y)
        
            # -------------------------------------------------
            # KEEP ONLY LONG ENOUGH SIGNALS
            # -------------------------------------------------
            if len(y) < N_target:
                continue
        
            # -------------------------------------------------
            # TRUNCATE to 4*Fs
            # -------------------------------------------------
            y = y[:N_target]
        
            Y_list.append(y)
        
        print(f"Number of valid signals: {len(Y_list)}")
        
        # -------------------------------------------------
        # STACK
        # -------------------------------------------------
        Y = np.stack(Y_list)   # (N, T)
        print("Y",Y.shape)
        W, N = Y.shape
        print("N",N,"W",W)
        # -------------------------------------------------
        # CREATE FAKE VARIABLES
        # -------------------------------------------------
        
        X = np.zeros_like(Y)#Y.copy()

        T, N = Y.shape
        
        grid = np.linspace(0, (N-1), N)/Fs
        
        G = np.repeat(grid[None, :], T, axis=0)
        G_prime = np.ones_like(G)
        
        print("X",X.shape)
        print("G",G.shape)
        
       
        """
        X = np.deppcopy(Y)
        G = np.array([np.linspace(0,(N-1)/N,N)]*T)
        G_prime = np.array([np.ones_like(Y)]*T)  
        print("G",G.shape)
        """
        # -------------------------------------------------
        # SPLIT 
        # -------------------------------------------------

        
        data_train = {
            "X": X,
            "Y": Y,
            "G": G,
            "G_prime": G_prime
        }
        
        data_val = {
            "X": X,
            "Y": Y,
            "G": G,
            "G_prime": G_prime
        }
        
        data_test = {
            "X": X,
            "Y": Y,
            "G": G,
            "G_prime": G_prime
        }
        
        # -------------------------------------------------
        # FINAL DICT (TON FORMAT)
        # -------------------------------------------------
        data_dict = {
            "X_train": data_train["X"],
            "Y_train": data_train["Y"],
            "G_train": data_train["G"],
            "G_prime_train": data_train["G_prime"],
            "X_val": data_val["X"],
            "Y_val": data_val["Y"],
            "G_val": data_val["G"],
            "G_prime_val": data_val["G_prime"],
            "X_test": data_test["X"],
            "Y_test": data_test["Y"],
            "G_test": data_test["G"],
            "G_prime_test": data_test["G_prime"]
        }
        
        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------
        
        self.save_dataset(filepath, data_dict)
        #np.savez("real_dataset.npz", **data_dict)
    
        print("Dataset saved.")
        
        return data_dict
            
    
#%%
if __name__ == "__main__":

   
    import matplotlib.pyplot as plt


    from Time_Warping_Functions import TimeWarping1D
    from Stationarity_Score import StationarityScore
    from CWT import WaveletTransform

    # =========================================================
    # PARAMETERS
    # =========================================================
    Fs = 44100
    T = 1
    N = int(np.ceil(Fs * T))
    
    file ="dataset.npz"
    
    type_gamma = ["gauss"]  # ["gauss", "cos", "chebyshev"]

    nb_win_train = 500
    nb_win_val   = 50
    nb_win_test  = 50

    # =========================================================
    # DATASET GENERATION 
    # =========================================================
    gen = GenerateDatabase(Fs)

    data = gen.generate_dataset(
        nb_win_train, T, Fs, type_gamma
    )

    X=data["X"]
    Y=data["Y"]
    G=data["G"]
    G_prime=data["G_prime"]
    


    # =========================================================
    # GENERATION & SAVE DATASET
    # =========================================================
    data = gen.generate_and_save_dataset(
        file,
        nb_train=nb_win_train,
        nb_val=nb_win_val,
        nb_test=nb_win_test,
        T=T,
        Fs=Fs,
        type_gamma=type_gamma,
        noise=False, 
        impulses=False
    )

    # =========================================================
    # LOAD DATASET
    # =========================================================
    data = gen.load_dataset(file)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    G_train = data["G_train"]
    G_prime_train = data["G_prime_train"]


    print("X_train shape", X_train.shape)
    
    # =========================================================
    # TOOLS
    # =========================================================
    TW = TimeWarping1D()
    StaScore = StationarityScore()

    t = np.linspace(0, N, N) / Fs

    # Stationarization / Unstationarization
    X_tilde = TW.stationarize_np(Y_train, G_prime_train)
    Y_tilde = TW.unstationarize_np(X_train, G_prime_train)

    print("X_tilde shape:", X_tilde.shape)


    # =========================================================
    # ANALYSIS MODULES
    # =========================================================
    fmin, fmax = 20, Fs / 2
    Ms = 100
    wavtyp = "dgauss"
    wavpar = 50

    wav = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
    
    
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    nb_display = 4

    for k in range(nb_display):

        x = X_train[k]
        y = Y_train[k]
        gamma = G_train[k]
        gamma_prime = G_prime_train[k]

        x_tilde = X_tilde[k]
        y_tilde = Y_tilde[k]
        
        

        StaScore = StationarityScore()
        TW1D = TimeWarping1D()

        # =========================================================
        # WAVELET TRANSFORMS
        # =========================================================
        W_x, t_axis_x, freqs_x = wav.cwt(x, wavpar)
        W_y, t_axis_y, freqs_y = wav.cwt(y, wavpar)



        # =========================================================
        # Stationarity scores
        # =========================================================
        
        _,score_x,score_y,_ = StaScore.score_TWET_np(np.abs(W_x),np.abs(W_y))
       
        
        # =========================================================
        # Spectra
        # =========================================================
        fx, Sx = wav.spectrum(x)
        fy, Sy = wav.spectrum(y)

        fx, Sx_tilde = wav.spectrum(x_tilde)
        fy, Sy_tilde = wav.spectrum(y_tilde)

        # =====================================================
        # PLOTS
        # =====================================================
    
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
        
        
        
    #%% Real database
    
    # -------------------------------------------------
    # PARAMETERS
    # -------------------------------------------------
    folder = "real_signals"
    file = "real_dataset.npz"
    Fs = 44100                # <-- à adapter
    N_target = 5  * Fs         # longueur fixe
    t=np.linspace(0,N_target-1,N_target)/Fs
    
    # =========================================================
    # SAVE real DATASET
    # =========================================================
    data2 = gen.save_real_dataset(folder,Fs,N_target,file)

    # =========================================================
    # LOAD DATASET
    # =========================================================
    data2 = gen.load_dataset(file)

    X_train = data2["X_train"]
    Y_train = data2["Y_train"]
    G_train = data2["G_train"]
    G_prime_train = data2["G_prime_train"]


    print("X_train shape", X_train.shape)
    
    fmin, fmax = 20, 3000
    Ms = 200
    dec = 100
    wavtyp = "sharp"
    wavpar = 50
    coeffs = 1024
    wav_real = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
        
    for k in range(len(X_train)):

        x = X_train[k]
        y = Y_train[k]
        gamma = G_train[k]
        gamma_prime = G_prime_train[k]

        StaScore = StationarityScore()
        TW1D = TimeWarping1D()

        # =========================================================
        # WAVELET TRANSFORMS
        # =========================================================
        
        W_x, t_axis_x, freqs_x = wav_real.cwt(x, wavpar)
        W_y, t_axis_y, freqs_y = wav_real.cwt(y, wavpar)

        # =========================================================
        # Stationarity scores
        # =========================================================
        
        _,score_x,score_y,_ = StaScore.score_TWET_np(np.abs(W_x),np.abs(W_y))
       
        # =========================================================
        # Spectra
        # =========================================================
        fx, Sx = wav.spectrum(x)
        fy, Sy = wav.spectrum(y)


        # =====================================================
        # PLOTS
        # =====================================================
    
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    
        # -------------------------
        # Signals
        # -------------------------
        axes[0, 0].plot(t[:coeffs], x[:coeffs], label=r"$x(t)$")
        axes[0, 0].set_title(r"Signal $x(t)$, score = {:.4f}".format(score_x))
        axes[0, 0].set_xlabel("Time (s)")   
        axes[0, 0].set_ylabel("Amp") 
        axes[0, 0].legend()
        axes[0, 0].grid()
    
        axes[0, 1].plot(t[:coeffs], y[:coeffs], label=r"$y(t)$")
        axes[0, 1].set_title(r"Signal $y(t)$, score = {:.4f}".format(score_y))
        axes[0, 1].set_xlabel("Time (s)")   
        axes[0, 1].set_ylabel("Amp") 
        axes[0, 1].legend()
        axes[0, 1].grid()
    

    
        # -------------------------
        # Spectra
        # -------------------------
        axes[1, 0].plot(fx[:N//2], Sx[:N//2], label="x")
        axes[1, 0].set_title("Spectrum X(f)")
        axes[1, 0].set_xlabel("Time (s)")   
        axes[1, 0].set_ylabel("Amp") 
        axes[1, 0].legend()
        axes[1, 0].grid()
    
        axes[1, 1].plot(fy[:N//2], Sy[:N//2], label="y")
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
        
    
    