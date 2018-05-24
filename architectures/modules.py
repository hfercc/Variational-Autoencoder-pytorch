"""
A file containing some basic building blocks 
"""

import torch.nn as nn


"""
A module which implements a stride-1 convolutional layer, then a stride-2 convolutional layer. 
The net effect of this is to cut the image size in half.

In addition, subsampling helps to increase the effective receptive field: 
  -- https://arxiv.org/pdf/1701.04128.pdf 
"""
def HalvingConv(in_channels, hidden_channels, out_channels):
  module = nn.Sequential(
    nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
    nn.Batchnorm2d(),
    nn.ReLU(),
    nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, stride=2),
    nn.Batchnorm2d(),
    nn.ReLU() 
  )
  return module

def DoublingConv(in_channels, hidden_channels, out_channels):
  module = nn.Sequential(
    nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=2),
    nn.Batchnorm2d(),
    nn.ReLU(),
    nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, padding=1, stride=1),
    nn.Batchnorm2d(),
    nn.ReLU()
  )

def MultiLayerConv(in_channels, hidden_channels, encoded_dims, num_layers=6):
  i = in_channels
  h = hidden_channels
  o = encoded_dims

  # the encoder
  encoder_modules = [HalvingConv(i, h, h)]
  for _ in range(num_layers-1):
    encoder_modules.append(HalvingConv(h,h,h))
  encoder_modules.append(HalvingConv(h,h,o))
  encoder_modules.append(Flatten())
  encoder_modules.append(nn.Linear(o,o))
  encoder = nn.Sequential(*encoder_modules)

  # the decoder 
  decoder_modules = [nn.Linear(o,o)]
  decoder_modules.append(Unflatten())
  decoder_modules.append(DoublingConv(o,h,h))
  for _ in range(num_layers-1):
    decoder_modules.append(DoublingConv(h,h,h))
  decoder_modules.append(DoublingConv(h,h,i))
  
  return encoder, decoder

