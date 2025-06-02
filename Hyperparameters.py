
from misc import *

cluster_path='./'  #Working repository

create_folder("Results/")

datasets_NAME='CIFAR10' #Other choices : "MNIST" or "FMNIST"  "CIFAR10"

taille_image=28  #Image_size
if datasets_NAME=='CIFAR10':
    taille_image=32



folder_results='Results'   #Folder to print results
num_block=20   
size_block=10  #corresponds to the structure of the latent code. Note : Latent_dim=num_block*size_block

operation="max"   #Other choice : "max" "sum" or "simple" (to choose a simple autoencoder, might fail on scripts designed specifically for Atomic autoencoders)
batch_size = 64  
learning_rate = 0.0001
epochs = 2   #Default 500 - may take some time
dataset_size=10 #Train dataset size Default 30000 half the dataset
subset_size = 10000   #Default 600
visualize_interval = 10  
blur=0.7

pretrained_model_file="Model_AAE/AAE_"+datasets_NAME+"_"+operation+".pt"     #Path to the pretrained model according to the desired operation and desired Dataset   


#/!\ Should not be modified if no model is trained
#Corresponds to maximum number of filter in a layer in the pretained model
#Depends on the type of dataset.
if datasets_NAME=="CIFAR10" or datasets_NAME=="FMNIST":
    complexity=32
else:
    complexity=8

