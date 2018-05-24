import torch
from architectures import *

encoder = E_TwoLayerFC()
decoder = D_TwoLayerFC()
input = torch.empty(10, 32, 32, 3, dtype=torch.float)
print(input.shape)
output = encoder(input)
print(output.shape)
final = decoder(output)
print(final.shape)
