import os
from datetime import date,datetime
import numpy as np
import torch




#===================================================================
# CUDA DEVICE SELECTION and associated functions
#===================================================================


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

#===================================================================
# PSNR
#===================================================================

def output_psnr_mse(img_orig, img_out):
    mse = np.mean((img_orig - img_out)**2)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

#===================================================================
# Folder creation to save models
#===================================================================
def create_folder(cluster_path):
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
        print("Folder "+cluster_path+" created")

def create_folder_training(cluster_path):
    date_model=str(date.today())
    newpath = cluster_path+"Model_n_Res/"+date_model

    if not os.path.exists(newpath):
        os.makedirs(newpath)
    else:
        k=1
        while os.path.exists(newpath):
            newpath = cluster_path+"Model_n_Res/"+date_model+"_"+str(k)
            k+=1
        os.makedirs(newpath)
    return newpath

#===================================================================
# IMPORT a model (AAE) to use
#===================================================================

def import_projection(file,device):
    projection=torch.jit.load(file,map_location=device)
    projection.eval()
    return projection
