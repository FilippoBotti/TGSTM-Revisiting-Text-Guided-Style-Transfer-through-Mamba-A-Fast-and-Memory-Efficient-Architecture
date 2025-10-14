import torch.nn as nn
from utility.ViT_helper import to_2tuple
from diffusers.models import AutoencoderKL
import torch
import einops
import math

class Net(nn.Module):
    def __init__(self, mamba, dec, encoder, args=None):
        super(Net, self).__init__()
        self.args = args
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) 
        self.mamba = mamba
        self.decode = dec
        self.mse_loss = nn.MSELoss()
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    
    def forward(self, content, style):
        B, C, H, W = content.shape
        content = self.encode(content)
   
        p,d,h,w = content.shape
        
        hs = self.mamba(style.float(),  None, content, None, None)      ##hs([4, 512, 28, 28])
        
        Ics = self.decode(hs)
            
        return Ics
