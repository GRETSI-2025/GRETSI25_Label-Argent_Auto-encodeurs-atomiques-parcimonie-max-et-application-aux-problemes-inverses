import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import torch.nn.parallel

from datetime import date,datetime
from Hyperparameters import *
from misc import *
from AAE_architecture import Autoencoder
from Datasets import generate_dataloader

torch.cuda.empty_cache()


device = get_default_device()


newpath=create_folder(cluster_path)

#===================================================================
# IMPORT the corresponding dataset
#===================================================================

train_loader,test_loader=generate_dataloader(subset_size,batch_size)
train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)


#===================================================================
# Building the model
#===================================================================

autoencoder = Autoencoder(num_block,size_block,operation,complexity,taille_image)
to_device(autoencoder,device)


criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=500, gamma=0.5)
# Training loop with visualization every 10 epochs

#=========================================================================
# Printing the Net architecture and the Hyperparameter in a external file
#=========================================================================

pytorch_total_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)

with open(newpath+"/Architecture_et_HP.txt", 'w') as f:
    f.write("Architure\n")
    f.write(str(autoencoder)+'\n')
    f.write("Number of parameter : "+str(pytorch_total_params)+"\n")
    f.write("Learning_rate : "+str(learning_rate)+"\n")
    f.write("Batch size : "+str(batch_size)+"\n")
    f.write("Number of epochs : "+str(epochs)+"\n")
    f.write("Dataset : "+datasets_NAME+"\n")
    f.write("Dataset Size : "+str(subset_size)+"\n")
    f.write("Operation of the AAE : "+operation+"\n")
    

#===================================================================
# Train Loop
#===================================================================

start_time=time.time()
losses=[]
val_losses=[] 
best_loss=1
best_epoch=0 
 
for epoch in range(epochs):
    autoencoder.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, _ = data
        optimizer.zero_grad()   
        torch.autograd.set_detect_anomaly(True)
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss/len(train_loader))

    autoencoder.eval()
    val_loss=0
    with torch.no_grad():
        for i, data in enumerate(test_loader,0):
            inputs,_=data
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            val_loss+=loss.item()
            if (val_loss<best_loss):
                model_scripted_best = torch.jit.script(autoencoder)
                best_loss=val_loss
                best_epoch=epoch
        val_losses.append(val_loss/len(test_loader))


    if epoch%visualize_interval==0:

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Validation Loss :{val_loss/len(test_loader)}')

#===================================================================
# Printing the visual results, the Losses and saving the model
#===================================================================

    if (epoch ) >= epochs-1:

        plt.figure(figsize=(24, 8))
        inputs_grid = inputs[:8].cpu().detach().numpy()
        outputs_grid = outputs[:8].cpu().detach().numpy()

        for j in range(8):
            plt.subplot(2, 8, j + 1)
            plt.imshow(inputs_grid[j][0], cmap='gray',vmin=0,vmax=1)
            plt.axis('off')

            plt.subplot(2, 8, j + 9)
            plt.imshow(outputs_grid[j][0], cmap='gray',vmin=0,vmax=1)
            plt.axis('off')
            psr=output_psnr_mse(inputs_grid[j][0],outputs_grid[j][0])
            plt.title("psnr="+f"{round(float(psr), 2):.{2}f}"+" dB")

        plt.savefig(newpath+'/Res_entrainement.png')
        plt.close()

# Save the trained model

model_scripted_best.save(newpath+'/'+operation+'-AAE_model.pt')



for i in range(0,len(losses)):
    losses[i]=float(np.log10(losses[i]))
plt.plot(losses)
plt.grid(True)
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(newpath+'/Loss_entrainement.png')
plt.close()
end_time=time.time()
print("Elapsed time ",end_time-start_time)

for i in range(0,len(val_losses)):
    val_losses[i]=float(np.log10(val_losses[i]))
plt.plot(val_losses)
plt.grid(True)
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(newpath+'/Loss_validation.png')
plt.close()

