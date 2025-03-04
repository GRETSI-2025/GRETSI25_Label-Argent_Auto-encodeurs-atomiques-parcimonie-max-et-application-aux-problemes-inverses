import torch.nn as nn
import torch


#===================================================================
# Atomic autoencoders architectures
#===================================================================


class Autoencoder(nn.Module):
    def __init__(self,num_blocks,size_blocks,operation,complexity,taille_image):
        self.taille_image=taille_image
        self.complexity=complexity
        self.num_block=num_blocks
        self.size_block=size_blocks
        self.operation=operation
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,int(self.complexity/2), kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            nn.Conv2d(int(self.complexity/2),self.complexity, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            nn.Flatten(),
            torch.nn.Linear(self.complexity*(int(self.taille_image/4)**2), 210),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(210, 200),
            torch.nn.LeakyReLU(),
        )
        self.mini_decoder=nn.Sequential(
            nn.Linear(self.size_block, 2*self.size_block),
            nn.LeakyReLU(),
            nn.Linear(2*self.size_block, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.complexity*(int(self.taille_image/4)**2)),
            )
        
        self.mini_decoder_deconv=nn.Sequential(
            nn.Conv2d(self.complexity,self.complexity, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.LeakyReLU(),
            nn.Conv2d(self.complexity,int(self.complexity/2), kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.LeakyReLU(),
            nn.Conv2d(int(self.complexity/2),1, kernel_size=3, stride=1, padding=1),
             )
        
        self.multscal=torch.nn.Parameter(torch.zeros(self.num_block))

    def forward(self, x):
        x = x.view(-1,1, self.taille_image, self.taille_image)
        x = self.encoder(x)
        x=x.view(-1,self.num_block,self.size_block)
        output=self.mini_decoder_deconv(self.mini_decoder(x[:]).view(-1,self.complexity,int(self.taille_image/4),int(self.taille_image/4)))

        output=output.view(-1,self.num_block,1,self.taille_image,self.taille_image)
        if self.operation=="sum":
            final=torch.sum(output,1)
        elif self.operation=="max":
            final,_=torch.max(output,1)
        else:
            final=output
        final=final.view(-1,1,self.taille_image,self.taille_image)
        return final
