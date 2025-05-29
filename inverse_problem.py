import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Dataset
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
# import sklearn
# from sklearn import datasets
from PIL import Image
import os
from Hyperparameters import *
from misc import *
from Datasets import generate_unique_dataloader
from Subsampling_operator import *
from Inverse_problem_hyperparameter import *



#===================================================================
# Perform the Projected gradient descend provided a projector
#===================================================================

def PGD(x_star,projection,seuil,k_max,mu_s,pad):
    y=mult_A(x_star,pad)
    x_0=np.random.rand(taille_image,taille_image)*0

    x=np.copy(x_0)
    res=mult_A(x,pad)-y
    k=0

    norme_min=10
    x_min=0
    
    while(np.linalg.norm(res)>seuil and k<k_max):
        x_1=x-mu_s[0]*mult_A_transpose(res,taille_image,pad)
        x_1=np.float32(x_1)
        x_1=torch.tensor(x_1)
        x_1=projection(x_1).detach().numpy()
        x_1=np.reshape(x_1,(taille_image,taille_image))
        res_1=mult_A(x_1,pad)-y

        for s in range(1,len(mu_s)):
            x_s=x-mu_s[s]*mult_A_transpose(res,taille_image,pad)
            x_s=np.float32(x_s)
            x_s=torch.tensor(x_s)
            x_s=projection(x_s).detach().numpy()
            x_s=np.reshape(x_s,(taille_image,taille_image))
            res_s=mult_A(x_s,pad)-y
            if np.linalg.norm(res_s)<np.linalg.norm(res_1):
                mu_bon=mu_s[s]
                x_1=np.copy(x_s)
                res_1=np.copy(res_s)
        x=np.copy(x_1)
        res=np.copy(res_1)
        Norme=np.linalg.norm(res)
        if Norme<norme_min:
            norme_min=Norme
            x_min=x
        k+=1
    return x_min


#===================================================================
# Displaying the visual results for Sup res 
#===================================================================

def Sup_res_on_multiple_images(inputs,n,projection):
    output_psnr=[]

    for i in range(n):
        x_min=PGD(inputs[i],projection,seuil,k_max,mu_s,pad)
        output_psnr.append(output_psnr_mse(inputs[i],x_min))

    print((np.mean(output_psnr),np.std(output_psnr)))



#===================================================================
# Returns the mean PSNR of PGD given a dataset (inputs)
# Check Inverse_Problem_hyperparameter to change the sample size
#===================================================================

def Sup_res_display_results(inputs,projection):
    outputs=[]
    for i in range(10):
        outputs.append(PGD(inputs[i],projection,seuil,k_max,mu_s,pad))

    plt.figure(figsize=(32, 8))
    for j in range(10):
        x=np.float32(np.reshape(inputs[j],(1,taille_image,taille_image)))
        x=torch.tensor(x)
        x=(np.reshape(projection(x).detach().numpy(),(taille_image,taille_image)))

        plt.subplot(3, 10, j + 1)
        plt.imshow(inputs[j],cmap="gray",vmin=0,vmax=1)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, 10, j + 11)
        plt.imshow(outputs[j],cmap="gray",vmin=0,vmax=1)
        plt.title("Reconstruction")
        plt.axis('off')
        psr=output_psnr_mse(inputs[j],outputs[j])
        plt.title("PGD="+f"{round(float(psr), 2):.{2}f}"+" dB")

        plt.subplot(3, 10, j + 21)
        plt.imshow(x,cmap="gray",vmin=0,vmax=1)
        plt.axis('off')
        psr=output_psnr_mse(inputs[j],x)
        plt.title("AE="+f"{round(float(psr), 2):.{2}f}"+" dB")

    plt.savefig(folder_results+"/PGD_"+operation+".png")
    plt.show()
    plt.close()


#===================================================================
# Test
#===================================================================

projection=import_projection(pretrained_model_file,torch.device("cpu"))


test_loader=generate_unique_dataloader(Number_of_images_to_recover,Number_of_images_to_recover)
inputs,_=next(iter(test_loader))
inputs=((inputs.detach().numpy()))[:,0]

Sup_res_display_results(inputs,projection)


#Sup_res_on_multiple_images(inputs,len(inputs),projection):