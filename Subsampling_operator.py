import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from Hyperparameters import *
import torch
from Inverse_problem_hyperparameter import blur

def mult_A(x,pad=2,blur=blur):
    n=len(x)
    fx=fft2(x)
    s=blur
    x_i = np.arange(n)
    ind = (x_i>n/2)
    x_i[ind] = x_i[ind]-n
    y_i = np.arange(n)
    ind = (y_i>n/2)
    y_i[ind] = y_i[ind]-n
    X,Y = np.meshgrid(x_i,y_i)
    gs = np.exp(-(X**2+Y**2)/(2*s**2))
    gs = gs/np.sum(gs)
    fgs = fft2(gs)
    ffx=ifft2(fgs*fx).real 
    return ffx[::pad,::pad]

  

#Multiply by Transpose A
def mult_A_transpose(x,taille_image_original,pad=2,blur=blur):
    res=np.zeros((taille_image_original,taille_image_original))
    res[::pad,::pad]=x
    res=fft2(res)
    n=taille_image_original
    x_i = np.arange(n)
    ind = (x_i>n/2)
    x_i[ind] = x_i[ind]-n

    y_i = np.arange(n)
    ind = (y_i>n/2)
    y_i[ind] = y_i[ind]-n
    X,Y = np.meshgrid(x_i,y_i)
    s=blur
    
    gs = np.exp(-(X**2+Y**2)/(2*s**2))
    gs = gs/np.sum(gs)
    fgs = fft2(gs)
    res=ifft2(res*fgs).real  
    return res

def bil_interp_sous_echan(x,pad=2):
    x=(mult_A(x,pad))
    x=torch.tensor(x)
    x=x.view(1,1,14,14)
    x=nn.Upsample(scale_factor=2, mode='bilinear')(x)
    x=x.view(28,28).numpy()
    return x