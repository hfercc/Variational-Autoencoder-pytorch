"""
A file containing some basic building blocks 
"""

import torch.nn as nn
from architectures import Flatten, Unflatten

"""
A module which implements a stride-1 convolutional layer, then a stride-2 convolutional layer. 
The net effect of this is to cut the image size in half.

In addition, subsampling helps to increase the effective receptive field: 
  -- https://arxiv.org/pdf/1701.04128.pdf 
"""
def HalvingConv(in_channels, hidden_channels, out_channels):
  module = nn.Sequential(
    nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(hidden_channels),
    nn.ReLU(),
    nn.Conv2d(hidden_channels, out_channels, kernel_size=2, padding=0, stride=2),
    nn.BatchNorm2d(out_channels),
    nn.ReLU() 
  )
  return module

def DoublingConv(in_channels, hidden_channels, out_channels):
  module = nn.Sequential(
    nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, padding=0, stride=2),
    nn.BatchNorm2d(hidden_channels),
    nn.ReLU(),
    nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU()
  )
  return module

def MultiLayerConv(in_channels, hidden_channels, out_channels, input_size, num_layers=7):
  i = in_channels
  h = hidden_channels
  o = out_channels
  c,h,w = input_size

  c_enc, h_enc, w_enc = o, h // 2**num_layers, w // 2**num_layers
  enc_shape = (-1, c_enc, h_enc, w_enc)
  e = c_enc * h_enc * w_enc
  # the encoder
  encoder_modules = [HalvingConv(i, h, h)]
  for _ in range(num_layers-2):
    encoder_modules.append(HalvingConv(h,h,h))
  encoder_modules.append(HalvingConv(h,h,o))
  encoder_modules.append(Flatten())
  encoder_modules.append(nn.Linear(e,e))
  encoder = nn.Sequential(*encoder_modules)

  # the decoder 
  decoder_modules = [nn.Linear(e,e)]
  decoder_modules.append(Unflatten(enc_shape))
  decoder_modules.append(DoublingConv(o,h,h))
  for _ in range(num_layers-2):
    decoder_modules.append(DoublingConv(h,h,h))
  decoder_modules.append(DoublingConv(h,h,i))
  decoder = nn.Sequential(*decoder_modules)

  print(encoder, decoder, e)
    
  return encoder, decoder, e

