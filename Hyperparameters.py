

cluster_path='./'  #Working repository


datasets_NAME='CIFAR10' #Other choice : "MNIST" or "FMNIST" 

taille_image=28  #Image_size
if datasets_NAME=='CIFAR10':
    taille_image=32



folder_results='Results'   #Folder to print results
num_block=20   
size_block=10  #corresponds to the structure of the latent code. Note : Latent_dim=num_block*size_block

operation="max"   #Other choice : "sum" or "simple" (to choose a simple autoencoder, might fail on scripts designed specifically for Atomic autoencoders)
batch_size = 64  
learning_rate = 0.0001
epochs = 2 
subset_size = 10
visualize_interval = 10  
blur=0.7

pretrained_model_file="Model_AAE/AAE_"+datasets_NAME+"_"+operation+".pt"     #Path to the pretrained model according to the desired operation and desired Dataset   


#/!\ Should not be modified if no model is trained
#Corresponds to maximum number of filter in a layer in the pretained model
#Depends on the type of dataset.
if datasets_NAME=="CIFAR10":
    complexity=32
else:
    complexity=8

