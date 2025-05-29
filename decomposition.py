import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Dataset
import matplotlib.pyplot as plt
import numpy as np

from Datasets import generate_unique_dataloader
from Hyperparameters import *
from misc import import_projection


#===================================================================
# Returns 1 or 0 whether the atom is activated or not
#===================================================================

def is_activated(atom,Norme_input,seuil):
    norme=np.linalg.norm(atom)
    return int(norme/Norme_input>=seuil)


#======================================================================
# Print histogram of number atoms plus 
# the evolution of the atoms norms in a decreasing order 
#======================================================================

def number_activated_atomes_and_norm(inputs,projection):
    inputs_norme=[np.linalg.norm(inputs[i].detach().numpy()) for i in range(len(inputs))]
    atoms_norm=np.array([[0. for j in range(20)] for i in range(len(inputs))])
    number_activated_atoms=[0 for i in range(len(inputs))]
    decoded=projection.encoder(inputs).view(-1,projection.num_block,projection.size_block)
    for j in range(len(inputs)):
        output=projection.mini_decoder_deconv(projection.mini_decoder(decoded[j,:]).view(-1,complexity,int(taille_image/4),int(taille_image/4))).view(-1,taille_image,taille_image)
        for k in range(len(output)):
            norme=np.linalg.norm(np.clip(output[k].detach().numpy(),0,1))
            number_activated_atoms[j]+=is_activated(norme,inputs_norme[k],0.05)
            atoms_norm[j,k]=norme
    atoms_norm=np.sort(atoms_norm)
    cumsum_atoms=np.array([np.cumsum(atoms_norm[j],dtype=float) for j in range(len(inputs))])
    Decroissance=np.array([np.mean(atoms_norm[:,j]) for j in range(20)])
    Decroissance=np.flip(Decroissance)
    fig, ax1 = plt.subplots()

    ax1.hist(number_activated_atoms, color='g')
    ax1.set_xlabel('Number of activated atoms',fontsize=15)
    ax1.set_ylabel('Number of images', color='g',fontsize=15)
    ax1.set_xlim(0,20)
    ax1.set_ylim(0,300)
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()
    ax2.plot(Decroissance, 'b-',label='Sorted Average norm of atoms ')
    ax2.set_label("Test")
    ax2.set_ylim(0,10)

    ax2.set_ylabel('Norm of atoms', color='b',fontsize=15)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.legend(fontsize=17)

    plt.savefig(folder_results+'/'+operation+"_"+datasets_NAME+"_Activated.png")
    plt.show()
    plt.close()

#===================================================================
# Displaying the decomposition of elements indx_inputs in inputs
#===================================================================

def affiche_decomposition_inputs(inputs,indx_inputs,projection):
    x=inputs[indx_inputs:indx_inputs+1]
    x=projection.encoder(x)
    x=x.view(-1,projection.num_block,projection.size_block)
    decomposition=[]
    decomposition.append(projection.mini_decoder_deconv(projection.mini_decoder(x[:,0]).view(-1,complexity,int(taille_image/4),int(taille_image/4))).view(taille_image,taille_image).detach().numpy())
    for i in range(1,projection.num_block):
        decomposition.append(projection.mini_decoder_deconv(projection.mini_decoder(x[:,i]).view(-1,complexity,int(taille_image/4),int(taille_image/4))).view(taille_image,taille_image).detach().numpy())
    plt.figure(figsize=(10, 8))
    n=projection.num_block
    ligne=int((n)/5)+1
    for j in range(5):
        for k in range(int(n/5)):
            plt.subplot(ligne, 5, j + 1+5*(k))
            plt.imshow(decomposition[j+5*k],cmap="gray",vmin=0,vmax=1)
            plt.axis('off')
    plt.subplot(ligne, 5, 21)
    plt.imshow(inputs[indx_inputs][0].detach().numpy(),cmap="gray",vmin=0,vmax=1)
    plt.axis('off')
    plt.savefig(folder_results+'/Decomposition_'+operation+"_"+datasets_NAME+'.png')
    plt.show()
    plt.close()



#===================================================================
# Testing
#===================================================================

test_loader=generate_unique_dataloader(subset_size,subset_size)
inputs,_=next(iter(test_loader))
projection=import_projection(pretrained_model_file,torch.device("cpu"))

affiche_decomposition_inputs(inputs,0,projection)

number_activated_atomes_and_norm(inputs,projection)
