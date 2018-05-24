import torch
import torch.nn as nn
import numpy as np

def E_TwoLayerFC():
    encoder = nn.Sequential(
        Flatten(),
        nn.Linear(32 * 32 * 3, 200),
        nn.ReLU(),
        nn.Linear(200, 50)
    )
    return encoder

def D_TwoLayerFC():
    decoder = nn.Sequential(
        nn.Linear(50, 200),
        nn.ReLU(),
        nn.Linear(200, 32 * 32 * 3), 
        Unflatten([-1, 3, 32, 32])
    )    
    return decoder

class Flatten(nn.Module):
    def forward(self, x):
        self.shape = x.shape
        return x.view(x.shape[0], -1)

class Unflatten(nn.Module):
    def __init__(self, shape):
        super(Unflatten, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)

def test_encoder(encoder):
    input = torch.from_numpy(np.zeros([10, 3, 32, 32]))
    output = encoder(x)
    print(out.shape) 
