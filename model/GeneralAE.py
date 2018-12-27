import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralAE(nn.Module):
    # hidden layer가 한개여서 activation은 안쓴다.
    def __init__(self):
        super(GeneralAE,self).__init__()
        self.encoder = nn.Linear(28*28,20)
        self.decoder = nn.Linear(20,28*28)
        self.tanh = nn.Tanh()
                
    def forward(self,x):
        batch_size = x.size()[0]
        x = x.view(batch_size,-1)
        latent_feature = self.encoder(x)
        out = self.tanh(self.decoder(latent_feature))
        out = out.view(batch_size,1,28,28)
                
        return out
    
class GeneralAE_vis(nn.Module):
    def __init__(self):
        super(GeneralAE_vis,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.ReLU()
        )
                
    def forward(self,x):
        batch_size = x.size()[0]
        x = x.view(batch_size,-1)
        latent_feature = self.encoder(x)
        y = self.decoder(latent_feature).view(batch_size,1,28,28)
                
        return y, latent_feature