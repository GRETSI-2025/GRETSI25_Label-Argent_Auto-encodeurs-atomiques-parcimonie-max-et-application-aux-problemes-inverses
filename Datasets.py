import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from Hyperparameters import datasets_NAME


#===================================================================
#Corresponds to  the transformation to apply following the dataset
#===================================================================


if datasets_NAME=="CIFAR10":
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


#===================================================================
# Creating Train and test dataloaders 
#===================================================================

def generate_dataloader(subset_size,batch_size,transform=transform,datapath=None):
    dl=False
    if not datapath:
        dl=True
        datapath=''
    if datasets_NAME=="CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data'+datapath, train=True, download=True, transform=transform)
    elif datasets_NAME=="FMNIST":
        full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
 
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    train_size = int(0.8 * len(subset_dataset))
    val_size = len(subset_dataset) - train_size
    train_dataset, _ = random_split(subset_dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    if datasets_NAME=="CIFAR10":
        test_dataset = datasets.CIFAR10(root='./data'+datapath, train=False, download=True, transform=transform)
    elif datasets_NAME=="FMNIST":
        test_dataset = datasets.FashionMNIST(root='./data'+datapath, train=False, download=True, transform=transform)
    else:
        test_dataset = datasets.MNIST(root='./data'+datapath, train=False, download=True, transform=transform)

    test_loader= DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_loader,test_loader


#===================================================================
# Creating test dataloaders for experiments
#===================================================================

def generate_unique_dataloader(subset_size,batch_size,transform=transform,datapath=None):
    dl=False
    if not datapath:
        dl=True
        datapath=''
    if datasets_NAME=="CIFAR10":
        test_dataset = datasets.CIFAR10(root='./data'+datapath, train=False, download=True, transform=transform)
    elif datasets_NAME=="FMNIST":
        test_dataset = datasets.FashionMNIST(root='./data'+datapath, train=False, download=True, transform=transform)
    else:
        test_dataset = datasets.MNIST(root='./data'+datapath, train=False, download=True, transform=transform)

    test_loader= DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader


