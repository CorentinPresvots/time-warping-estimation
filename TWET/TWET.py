# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 15:45:53 2025

@author: coren
"""


import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np
import scipy.interpolate
 
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




# =========================================================
# Dilated Conv Block
# =========================================================
class DilatedBlock2D(nn.Module):
    """
    2D dilated convolution block (time-frequency)
    Dilation is applied ONLY along time axis.
    """
    def __init__(self, C_in, C, kernel_size, dilation):
        super().__init__()

        kf, kt = kernel_size

        self.conv = nn.Conv2d(
            C_in, C,
            kernel_size=(kf, kt),
            dilation=(1, dilation),
            padding=(kf // 2, (kt // 2) * dilation)
        )

        self.norm = nn.LayerNorm(C)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x)

        # LayerNorm over channels
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)

        return x + self.act(out)
    
    
    
    

class Model(nn.Module,TimeWarping1D,TimeWarping2D,StationarityScore):
    def __init__(self,Fs, N, C, F, T,fmin,fmax):
        nn.Module.__init__(self)
        TimeWarping1D.__init__(self)
        TimeWarping2D.__init__(self)
        StationarityScore.__init__(self)


        self.Fs = Fs # fréquence d'échantillonnage
        
        self.N = N # Nombre d'échantillon du signal original
        self.C = C # Nombre de waveparam channel
        self.F = F # Nombre de fréquence scale
        self.T = T # Nombre d'échantillon de T
        self.dec = int(N/T) # facteur de décimation
        
        
  
        self.fmin=fmin   # ne sert que pour Time_warping 2D pour le décalage 
        self.fmax=fmax
        
      
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print("self.device",self.device)
        
        # -------------------
        # Convolution stack
        # -------------------
        K           = 32 # nombre de filtre
        kernel_size = (3,3)
        pading_size=(kernel_size[0]//2,kernel_size[1]//2)
        self.down   = nn.Conv2d(self.C,K, kernel_size=kernel_size, stride=(1,1), padding=pading_size)
        
      
        self.conv1 = DilatedBlock2D(K,K,kernel_size,1)
        self.conv2 = DilatedBlock2D(K,K,kernel_size,2) 
        self.conv3 = DilatedBlock2D(K,K,kernel_size,4) 
        self.conv4 = DilatedBlock2D(K,K,kernel_size,8) 
        self.conv5 = DilatedBlock2D(K,K,kernel_size,16) 
        self.conv6 = DilatedBlock2D(K,K,kernel_size,32) 
        self.conv7 = DilatedBlock2D(K,K,kernel_size,64)        

  
        self.channel_proj = nn.Conv2d(
            in_channels=K,
            out_channels=1,
            kernel_size=kernel_size,
            padding=pading_size
        )

        self.freq_proj = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(self.F,kernel_size[1]),
            padding=(0,pading_size[1])
        )
        
        self.basis = None

    # ---------------------------
    # Utility math functions
    # ---------------------------
    def bspline_basis(self, N, k, degree=3):
        # Points d'évaluation
        #Base de spline pour l'interpolation
        x = np.linspace(0, 1, N)
    
        # Nœuds avec répétition aux bornes
        n_knots = k + degree + 1
        knots = np.linspace(0, 1, n_knots - 2 * degree)
        knots = np.concatenate((
            np.zeros(degree),
            knots,
            np.ones(degree)
        ))
    
        basis = np.zeros((N, k))
        for i in range(k):
            c = np.zeros(k)
            c[i] = 1
            b = scipy.interpolate.BSpline(knots, c, degree)(x)
            basis[:, i] = b
    
        return torch.from_numpy(basis).float().to(self.device)


    
    def spline_smoothness_loss(self, coeffs):
        # coeffs: (B, k)
    
        diff1 = coeffs[:, 1:] - coeffs[:, :-1]          # dérivée 1
        diff2 = diff1[:, 1:] - diff1[:, :-1]            # dérivée 2 (curvature)
        diff3 = diff2[:, 1:] - diff2[:, :-1]
        diff4 = diff3[:, 1:] - diff3[:, :-1]
        diff5 = diff4[:, 1:] - diff4[:, :-1]

    
        loss = (
            0 * torch.mean(diff1**2, dim=1) +
            1 * torch.mean(diff2**2, dim=1) +
            0 * torch.mean(diff3**2, dim=1) +
            0 * torch.mean(diff4**2, dim=1) +
            0 * torch.mean(diff5**2, dim=1)
        )
    
        return loss.unsqueeze(1)




    # ---------------------------
    # Loss wrapper
    # ---------------------------
    def loss_fn(self, y_true,y_wav_true, out_pred,gamma_prime_pred_true=0):
        """
        y_true: (B, N)
        out_pred: (B, 3, N)
        """
        
        gamma_prime_pred = out_pred["gamma_prime_pred"]
        x_wav_pred = out_pred["x_wav_pred"]
        

        # stationarity score
        scor_stationarity,_,_,_ =  self.score_TWET_th(x_wav_pred,y_wav_true)
       
        loss_coeff_smooth = self.spline_smoothness_loss(gamma_prime_pred[:, ::self.dec])
        
        loss = torch.mean(
           scor_stationarity + 10 *loss_coeff_smooth )
       
        return loss

    

    def normalize(self, y):
        # y: (B, 1, kf,kt)
        
        #print("y",y.shape)
        std =  torch.std(y, dim=(2,3), keepdim=True) + 1e-10
        
        #print("std",std.shape)
        mean = torch.mean(y, dim=(2,3), keepdim=True)
        
        return (y - mean) / std, mean, std
    

 
             

    
    def check_input_y(self,input_y):
        # Accepté :
        # (B, N) ou (B, 1, N)
    
        if input_y.ndim == 2:
            input_y = input_y.unsqueeze(1)   # (B,1,N)
        elif input_y.ndim == 3:
            assert input_y.shape[1] == 1, "input_y must be (B,1,N)"
        else:
            raise ValueError(f"input_y shape invalide: {input_y.shape}")
    
        return input_y
   
    def check_input_y_wav(self, input_y_wav):
    
       
        # Cas (B, kf, kt)
        if input_y_wav.ndim == 3:
            input_y_wav = input_y_wav.unsqueeze(1) #(B, 1,kf, kt)
       
        elif input_y_wav.ndim == 4:
            pass
        else:
            raise ValueError(
                f"input_y_wav shape invalide: {input_y_wav.shape}. "
                "Expected (B,self.n_channels,kf,kt)"
            )
    

        return input_y_wav
        
        

    def forward(self, input_y,input_y_wav):
        # input_y (B,N) ou (N,) Signal brute 
        # input_y_wav (B,C,F,T) ou (B,F,T) ou (B,F,T) où (F,T) et doit sortir de la forme (B,C,F,T) 
        
        #input_y = self.check_input_y(input_y)
        input_y_wav = self.check_input_y_wav(input_y_wav)
     
        
        # --- Normalisation ---
        y_norm, mean, std = self.normalize(input_y_wav)
        #print("y_norm (B,C,F,T)",y_norm.shape)
        
        y_down= self.down(y_norm)
        #print("y_down (B, C, F,T)", y_down.shape)
        
        # Conv stack
        out = self.conv1(y_down)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        
        #out = F.dropout(out, p=0.1, training=self.training)
         
        latent =  self.channel_proj(out)  
        #print("latent (B,1,F,T)",latent.shape)
        
        self.out = Fun.softplus(latent.squeeze(1))
        #print("self.out (B,F,T)",self.out.shape)
  
        latent_proj=self.freq_proj(latent)
        #print("latent_proj (B,1,1,T)",latent_proj.shape)
        
        coeffs = Fun.softplus(latent_proj)
        coeffs = coeffs.squeeze(1,2)
        #print("coeffs (B,T)",coeffs.shape)
       
        coeffs /= (torch.mean(coeffs, dim=1, keepdim=True) + 1e-8)#*(self.N/self.Fs)
     
        
        if self.basis is None or self.basis.shape[1] != coeffs.shape[1]:
            self.noeud = coeffs.shape[1]
            print("self.noeud",self.noeud)
            self.basis = self.bspline_basis(self.N, coeffs.shape[1],degree=1).to(coeffs.device)
        #print("self.basis.shape",self.basis.shape)    
        

        # We need (B, N, 1) = (1, N, k) @ (B, k, 1) via broadcasting
        gamma_prime_pred = torch.matmul(self.basis, coeffs.unsqueeze(-1)).squeeze(-1)  
        #print("gamma_prime_pred (B, N) ",gamma_prime_pred.shape)


        #print("input_y (B, N)",input_y.shape)
        x_pred = self.stationarize_th(input_y, gamma_prime_pred)
        
        
        #print("coeffs",coeffs.shape,"input_y_wav",input_y_wav.shape)
        x_wav_pred = self.estimSpectrum_th(input_y_wav,self.Fs,self.fmin,self.fmax,self.F,coeffs)
        #print("x_wav_pred (B, F, T)",x_wav_pred.shape)
        
        outputs = {
            "x_pred": x_pred,                      # (B, N)
            "gamma_prime_pred": gamma_prime_pred,  # (B, N)
            "x_wav_pred": x_wav_pred        # (B, F, N)
        }
        
       
        return outputs



        


    
#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
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
    real= True
    if real == False :
        file = "dataset.npz"
    else :
        file = "real_dataset.npz" 
    gen = GenerateDatabase()
    
    file_path = os.path.join(db_path, file)
    
    data = gen.load_dataset(file_path)
    
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    G_train = data["G_train"]
    G_prime_train = data["G_prime_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    G_val = data["G_val"]
    G_prime_val = data["G_prime_val"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    G_test = data["G_test"]
    G_prime_test = data["G_prime_test"]
    
    
    
    # =========================================================
    # ANALYSIS MODULES
    # =========================================================
    Fs = 44100
    
   
    if real == False :
        fmin, fmax = 20, Fs / 2
    else :
        fmin, fmax = 200, 3000
    N=X_train.shape[1]
    Ms = 50
    dec = 100
    T = int(N/dec) # nuombre d'échantillon apres décimation
    wavtyp = "sharp"# "sharp"|"dgauss"
    wavpars = [50]#[4,8,12,16,32,64,128]

    wav = WaveletTransform(Fs, fmin, fmax, Ms, wavtyp)
    
    
    # =========================================================
    # WAVELET TRANSFORMS
    # =========================================================
    X_wav_train, t_axis_x_train, freqs_x_train = wav.cwt(X_train, wavpars,dec)
    Y_wav_train, t_axis_y_train, freqs_y_train = wav.cwt(Y_train, wavpars,dec)
    
    X_wav_val, t_axis_x_val, freqs_x_val = wav.cwt(X_val, wavpars,dec)
    Y_wav_val, t_axis_y_val, freqs_y_val = wav.cwt(Y_val, wavpars,dec)
    
    X_wav_test, t_axis_x_test, freqs_x_test = wav.cwt(X_test, wavpars,dec)
    Y_wav_test, t_axis_y_test, freqs_y_test = wav.cwt(Y_test, wavpars,dec)
    
    X_wav_train=np.abs(X_wav_train)
    Y_wav_train=np.abs(Y_wav_train)
    
    X_wav_val=np.abs(X_wav_val)
    Y_wav_val=np.abs(Y_wav_val)
    
    X_wav_test=np.abs(X_wav_test)
    Y_wav_test=np.abs(Y_wav_test)
    
 
    print("X_wav_train",X_wav_train.shape)
    print("X_wav_val",X_wav_val.shape) 
    print("X_wav_test",X_wav_test.shape) 
    
    
        
    #%%
    # ---------------------
    # 2. Conversion en tenseurs PyTorch
    # ---------------------
    import gc
    from torch.utils.data import TensorDataset,DataLoader
             
    gc.collect()
    torch.cuda.empty_cache()  
    
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    
    Y_wav_train_t = torch.tensor(Y_wav_train, dtype=torch.float32)
    Y_wav_val_t = torch.tensor(Y_wav_val, dtype=torch.float32)
    
    G_prime_train_t = torch.tensor(G_prime_train, dtype=torch.float32)
    G_prime_val_t = torch.tensor(G_prime_val, dtype=torch.float32)
    
    # Dataset et DataLoader
    train_ds = TensorDataset(Y_train_t, Y_train_t,Y_wav_train_t,G_prime_train_t)
    val_ds = TensorDataset(Y_val_t, Y_val_t,Y_wav_val_t,G_prime_val_t)


    print("Y_train_t",Y_train_t.shape)

     
   
    #%%
    # ---------------------
    # 3. Initialisation du modèle
    # ---------------------
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights_path = "v1.pth"
    train_model = True
    retrain_model= False
    
    batch_size =  50
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    
    model = Model(Fs=Fs,N=N,C=len(wavpars),F=Ms,T=T,fmin=fmin,fmax=fmax).to(device)  #.to(device)
    
    if os.path.exists(weights_path) and retrain_model:
        print(f"🔄 Chargement des poids depuis {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))

    # --------------------- 
    # 4. Entraînement
    # ---------------------
    nb_epochs = 800
   
    optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-3,
    #weight_decay=1e-4,
    #betas=(0.9, 0.99),
    #eps=1e-8
    )
    
    
   
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    if train_model:
        for epoch in range(nb_epochs):
            # 🔹 Entraînement
            model.train()
            running_loss = 0.0
            for inputs, targets,y_wav_train,g_primes in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                y_wav_train = y_wav_train.to(device)
                g_primes = g_primes.to(device)

                optimizer.zero_grad()
                outputs = model(inputs,y_wav_train)
                loss    = model.loss_fn(targets,y_wav_train,outputs,g_primes)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)

            # 🔹 Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets,y_wav_val,g_primes in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    y_wav_val = y_wav_val.to(device)
                    g_primes = g_primes.to(device)
                    
                    outputs = model(inputs,y_wav_val)
                    loss    = model.loss_fn(targets,y_wav_val,outputs,g_primes)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)

            train_losses.append(epoch_loss)
            val_losses.append(val_loss)


            # 🔹 Sauvegarde du meilleur modèle
            if epoch%10==0:
                print(f"Epoch [{epoch+1}/{nb_epochs}] - Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")
                #"""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), weights_path)
                    print(f"✅ Nouveau meilleur modèle sauvegardé ({weights_path})")
                """
       
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), weights_path)
                    #if epoch%10==0:
                    #print(f"Epoch [{epoch+1}/{nb_epochs}] - Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")
                    print(f"✅ Nouveau meilleur modèle sauvegardé ({weights_path})")
            
                """

        
        # 🔹 Affichage des courbes de perte (échelle logarithmique)
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses[10:], label='Training Loss')
        plt.plot(val_losses[10:], label='Validation Loss')
        plt.title("Évolution de la perte au cours de l'entraînement (échelle log)")
        plt.xlabel("Époque")
        plt.ylabel("MSE Loss (log)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.7)
        
        # 🔹 Passage en échelle logarithmique sur l’axe Y
        plt.yscale("log")
        # 🔹 Fixe les bornes de l'axe Y
        #plt.ylim(0, 20)
        
        plt.tight_layout()
        plt.show()
        
        gc.collect()
        torch.cuda.empty_cache()    
    # ---------------------
    # 5. Chargement du meilleur modèle
    # ---------------------
    model.load_state_dict(torch.load(weights_path, map_location=device,weights_only=False))
    model.eval()
    

    
   
    
    #%%
    # ---------------------
    # 5. Création base de test
    # ---------------------
    nb_win_used = int(np.min([50,X_test.shape[0]]))
    channel_used = -1#np.min([-1,len(wavpars)-1])
        
    Y_used = np.copy(Y_test[:nb_win_used])
    Y_wav_used = np.copy(Y_wav_test[:nb_win_used])
    X_used = np.copy(X_test[:nb_win_used])
    X_wav_used=np.copy(X_wav_test[:nb_win_used])
    G_used = np.copy(G_test[:nb_win_used])
    G_prime_used = np.copy(G_prime_test[:nb_win_used])
    
    t_axis_x_used = t_axis_x_test
    freqs_x_used = freqs_x_test
    
    
    

        
    X_wav_used_torch =torch.tensor(X_wav_used, dtype=torch.float32).to(device)
    
    Y_used_torch =torch.tensor(Y_used, dtype=torch.float32).to(device)
    Y_wav_used_torch =torch.tensor(Y_wav_used, dtype=torch.float32).to(device)
    
    X_used_torch =torch.tensor(X_used, dtype=torch.float32).to(device)
    G_prime_used_torch =torch.tensor(G_prime_used, dtype=torch.float32).to(device)
    
    import time
    start = time.time()
    # Mode évaluation + pas de gradient
    model.eval()
    with torch.no_grad():
        out_pred_t = model(Y_used_torch,Y_wav_used_torch)
        # out_pred_t shape: (B, 3, N)
    end = time.time()
    
    print(f"Elapsed time per signal: {(end - start)/(nb_win_used):.4f} s")
    
        
    # Récupérer les sorties et repasser en numpy
    X_pred_used_torch = out_pred_t["x_pred"]
  
    G_prime_pred_used_torch = out_pred_t["gamma_prime_pred"]
    X_wav_pred_used_torch = out_pred_t["x_wav_pred"]
    
    X_pred_used = X_pred_used_torch.cpu().numpy()
   
    G_prime_pred_used = G_prime_pred_used_torch.cpu().numpy()
    X_wav_pred_used = X_wav_pred_used_torch.cpu().numpy()



    
    print("<MSE> (gamma'_pred-gamma')={:.8f}".format(np.mean(np.mean((G_prime_pred_used - G_prime_used)**2, axis=1))))
    print("<MAE> (gamma'_pred-gamma')={:.8f}".format(np.mean(np.abs(G_prime_pred_used - G_prime_used))))
    
    
    coefs_tronc = int(0.01*N)
    print(r"<MSE> (gamma'_pred-gamma') tronc={:.8f}".format(np.mean(np.mean((G_prime_pred_used[:,coefs_tronc:N-coefs_tronc] - G_prime_used[:,coefs_tronc:N-coefs_tronc] )**2, axis=1))))
    print(r"<MAE> (gamma'_pred-gamma') tronc={:.8f}".format(np.mean(np.abs(G_prime_pred_used[:,coefs_tronc:N-coefs_tronc] - G_prime_used[:,coefs_tronc:N-coefs_tronc]))))
    
    
    
    
    
    
    
    X_wav_pred_used_torch =torch.tensor(X_wav_pred_used, dtype=torch.float32).to(device)

    Score_X_Y,Score_X,Score_Y,Stack=model.score_TWET_th(X_wav_pred_used_torch,Y_wav_used_torch)#.cpu().numpy()
    
    Score_X_Y=Score_X_Y.cpu().numpy()
    Score_X=Score_X.cpu().numpy()
    Score_Y=Score_Y.cpu().numpy()
    Stack=Stack.cpu().numpy()
    
    for w in range(len(wavpars)) : 
        print("wavpar = {}".format(wavpars[w]))
        print("Score X pred = {:.5f}".format(np.mean(Score_X[:,w])))
        
        print("Score Y = {:.5f}".format(np.mean(Score_Y[:,w])))
        
        print("score X / score Y = {:.5f}".format(np.mean(Score_X[:,w]/Score_Y[:,w])))
        
        #print("Stack= {:.5f}".format(np.mean(Stack[:,w])))
        
        print("----")
   # print("Score_X_Y= {:.5f}".format(np.mean(Score_X_Y)))
        
    
    
    
    from JEFAS import JEFAS
    
    J=JEFAS()
    Score_X_JEFAS=[]
    Score_Y_JEFAS=[]
    
    
    
    for id_sig in range(nb_win_used):
        score_X_JEFAS=[]
        score_Y_JEFAS=[]
        for w in range(len(wavpars)):
            score_X_JEFAS_w = J.loss_JEFAS(Y_used[id_sig], G_prime_pred_used[id_sig], wavparam=wavpars[w], freqmin=fmin/Fs, freqmax=fmax/Fs,Nf=Ms,Dt=dec)
            score_Y_JEFAS_w = J.loss_JEFAS(Y_used[id_sig], np.ones(N), wavparam=wavpars[w], freqmin=fmin/Fs, freqmax=fmax/Fs,Nf=Ms,Dt=dec)
            
            score_X_JEFAS.append(score_X_JEFAS_w)
            score_Y_JEFAS.append(score_Y_JEFAS_w)
        Score_X_JEFAS.append(score_X_JEFAS)
        Score_Y_JEFAS.append(score_Y_JEFAS)    
    Score_X_JEFAS=np.array(Score_X_JEFAS)  
    Score_Y_JEFAS=np.array(Score_Y_JEFAS) 
    
    
    #Score_X_JEFAS,_=model.score_JEFAS_simple(X_wav_pred_used,Y_wav_used,G_prime_pred_used[:,::dec])#.cpu().numpy()
    #Score_Y_JEFAS,_=model.score_JEFAS_simple(X_wav_pred_used,Y_wav_used,np.ones(np.shape(G_prime_pred_used[:,::dec])))#.cpu().numpy()
    
    
    
    
    for w in range(len(wavpars)) : 
        print("wavpar = {}".format(wavpars[w]))
        print("Score X pred JEFAS      = {:.5f}".format(np.mean(Score_X_JEFAS[:,w])))
        
        print("Score Y JEFAS           = {:.5f}".format(np.mean(Score_Y_JEFAS[:,w])))
        
        print("score Y / score X JEFAS = {:.5f}".format(np.mean(Score_Y_JEFAS[:,w]/Score_X_JEFAS[:,w])))
        

        print("----")



        




  

     
    
    #%% test de reconstruction
    
    from Stationarity_Score import StationarityScore
    from scipy.integrate import cumulative_trapezoid
    
    StaScore = StationarityScore()
    
    # =========================================================
    # Cumpute Scores np and torch
    # =========================================================
    _,score_x_np,score_y_np,score_x_y=StaScore.score_TWET_np(X_wav_pred_used,Y_wav_used)
    

    _,score_x_th,score_y_th,_=StaScore.score_TWET_th(X_wav_pred_used_torch, Y_wav_used_torch)#
    
    score_x_th=score_x_th.cpu().numpy()
    score_y_th=score_y_th.cpu().numpy()
    
    t=np.linspace(0,N-1,N)/Fs
    coeffs = 400
    for k in range(np.min([4,nb_win_used])) : 
        print("score_x_np = {:.8f}".format(score_x_np[k,channel_used,0]))
        print("score_x_th = {:.8f}".format(score_x_th[k,channel_used,0]))
       
        x=X_used[k]
        y=Y_used[k]
        
        x_pred = X_pred_used[k]
        
        score_x_pred = score_x_np[k,channel_used,0]
        score_x_pred_JEFAS = Score_X_JEFAS[k,channel_used]
        score_y=score_y_np[k,channel_used,0]
        score_y_pred_JEFAS = Score_Y_JEFAS[k,channel_used]
        
        # =========================================================
        # SPECTRAL ANALYSIS
        # =========================================================
        fx, Sx = wav.spectrum(X_used[k])
        fy, Sy = wav.spectrum(Y_used[k]) 
        
        fx_pred, Sx_pred = wav.spectrum(X_pred_used[k])
        

        gamma_prime= G_prime_used[k]
        gamma_prime_pred= G_prime_pred_used[k]
        
        gamma= G_used[k]
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
        axes[0, 0].set_title(r"Signal $\widetilde{{x}}(t)$, score TWET = {:.4f}, JEFAS = {:.4f}".format(score_x_pred,score_x_pred_JEFAS))
        axes[0, 0].set_xlabel("Time (s)")   
        axes[0, 0].set_ylabel("Amp")
        axes[0, 0].legend()
        axes[0, 0].grid()
    
        axes[0, 1].plot(t[:coeffs], y[:coeffs], label=r"$y(t)$")
        axes[0, 1].set_title(r"Signal $y(t)$, score TWET = {:.4f}, JEFAS = {:.4f}".format(score_y,score_y_pred_JEFAS))
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
       
        axes[4, 0].pcolormesh(t_axis_x_used,freqs_x_used, X_wav_pred_used[k,channel_used], shading="auto")
        axes[4, 0].set_title(r"$W_x$")
        axes[4, 0].set_xlabel("Time (s)")   
        axes[4, 0].set_ylabel("Freq (Hz)") 
    
        axes[4, 1].pcolormesh(t_axis_x_used,freqs_x_used, Y_wav_used[k,channel_used], shading="auto")
        axes[4, 1].set_title(r"$W_y$")
        axes[4, 1].set_xlabel("Time (s)")   
        axes[4, 1].set_ylabel("Freq (Hz)") 
    
        # =========================================================
        # DISPLAY
        # =========================================================
        plt.tight_layout()
        plt.show()
        
        
    
    
   
        

    
    
    
    

    
    
    #%% test importance du wavpar 
        
        
    # =========================================================
    # Study of wavelet parameter importance
    # =========================================================
    
    MSE_wavepar = []
    n_wavepar = len(wavpars)
    

    fig, axes = plt.subplots(n_wavepar, 2, figsize=(12, 3 * n_wavepar))
    # Force axes to be 2D
    axes = np.atleast_2d(axes)
    
    
    u = 0  # index of example to visualize
    
    for channel in range(n_wavepar):
    
        # -----------------------------------------------------
        # Keep only ONE wavelet channel (ablation study)
        # -----------------------------------------------------
        Y_wav_test_ = Y_wav_used_torch.clone()
        #Y_wav_test[:, :channel, :, :] = 0
        Y_wav_test_[:, channel + 1:, :, :] = 0
    
        # -----------------------------------------------------
        # Forward pass (no gradient)
        # -----------------------------------------------------
        model.eval()
        with torch.no_grad():
            outputs = model(Y_used_torch, Y_wav_test_)
    
        gamma_pred = outputs["gamma_prime_pred"].cpu().numpy()  # (B, N)
    
        # -----------------------------------------------------
        # Compute MSE between prediction and ground truth
        # -----------------------------------------------------
        #coefs_tronc = int(0.0 * N)
    
        mse = np.mean(
            (gamma_pred[:, coefs_tronc:N - coefs_tronc]
             - G_prime_used[:, coefs_tronc:N - coefs_tronc]) ** 2,
            axis=1
        )
    
        MSE_wavepar.append(mse)
    
        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------
        

        ax1 = axes[channel, 0]
        ax2 = axes[channel, 1]
    
        # --- Gamma comparison
        ax1.plot(gamma_pred[u][coefs_tronc:-coefs_tronc], label="Predicted γ'")
        ax1.plot(G_prime_used[u][coefs_tronc:-coefs_tronc], label="True γ'")
        ax1.set_title(f"Wavelet channel {channel}")
        ax1.legend()
        ax1.grid()
    
        # --- Squared error
        err = (gamma_pred[u][coefs_tronc:-coefs_tronc] - G_prime_used[u][coefs_tronc:-coefs_tronc]) ** 2
        ax2.plot(err)
        ax2.set_title(f"Squared error (MSE={mse[u]:.3e})")
        ax2.grid()
    
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # Convert list → array (B, C)
    # ---------------------------------------------------------
    MSE_wavepar_array = np.stack(MSE_wavepar, axis=1)
    
    # ---------------------------------------------------------
    # Plot MSE vs wavelet parameter
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(wavpars, MSE_wavepar_array[u], marker='o')
    plt.yscale("log")
    plt.xlabel("Wavelet parameter")
    plt.ylabel("MSE")
    plt.title("Impact of wavelet parameter on γ' prediction")
    plt.grid(True, which="both", linestyle="--")
    plt.show()
    
    
    # =========================================================
    # Full model analysis (latent + correlation)
    # =========================================================
    
    with torch.no_grad():
        _ = model(Y_used_torch.to(device), Y_wav_used_torch.to(device))
    
    latent_out = model.out.detach().cpu().numpy()  # (B, F, T)
    
    n_examples = min(10, len(Y_used))
    
    fig, axes = plt.subplots(
        n_examples, 4,
        figsize=(18, 2 * n_examples),
        gridspec_kw={'width_ratios': [1, 1, 1, 1]}
    )
    
    if n_examples == 1:
        axes = axes[None, :]
    
    for i in range(n_examples):
    
        # -----------------------------------------------------
        # 1. Real wavelet transform
        # -----------------------------------------------------
        pcm = axes[i, 0].pcolormesh(Y_wav_used[i, channel_used], cmap="viridis")
        axes[i, 0].set_title(f"Real wavelet (sample {i})")
        fig.colorbar(pcm, ax=axes[i, 0])
    
        # -----------------------------------------------------
        # 2. Latent representation
        # -----------------------------------------------------
        pcm = axes[i, 1].pcolormesh(latent_out[i], cmap="viridis")
        axes[i, 1].set_title("Latent representation")
        fig.colorbar(pcm, ax=axes[i, 1])
    
        # -----------------------------------------------------
        # 3. Gamma comparison
        # -----------------------------------------------------
        scores = score_x_y[i].reshape(len(wavpars))
    
        axes[i, 2].plot(G_prime_used[i][coefs_tronc:-coefs_tronc], label="True γ'")
        axes[i, 2].plot(G_prime_pred_used[i][coefs_tronc:-coefs_tronc], label="Predicted γ'")
        axes[i, 2].set_title(f"Best wavpar = {wavpars[np.argmin(scores)]}")
        axes[i, 2].legend()
        axes[i, 2].grid()
    
        # -----------------------------------------------------
        # 4. MSE vs wavelet parameter
        # -----------------------------------------------------
        mse_plot = MSE_wavepar_array[i]
    
        #axes[i, 3].plot(wavpars, mse_plot, marker='+')
        #axes[i, 3].plot(wavpars, 10e-3*np.ones(len(wavpars)))
        axes[i, 3].plot(wavpars, scores,'*-')
        axes[i, 3].set_yscale("log")
        axes[i, 3].set_xlabel("wavpars")
        axes[i, 3].set_ylabel("Amp")
        axes[i, 3].set_title("Score stationarity")
        axes[i, 3].grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # Memory cleanup (useful for GPU)
    # ---------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()    
        
    
    
    
    
   