import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import csv
from torch.utils.data import Dataset

class ImageDataset(Dataset):
  """
  Params:
  @dir_path: The path to the directory containing the data files
  @indices: A list of indices specifying which files are in the dataset. 
  """
  def __init__(self, dir_path, indices, transform=None):
    self.dir_path = dir_path
    self.indices = indices
    self.transform = transform

    self.labels = dict()
    with open(dir_path + "labels.csv", 'r') as csvfile:
      r = csv.reader(csvfile)
      i = 0
      for line in r:
        if line[-1] == "\n":
          line = line[:-1]
        self.labels[i] = line
        i += 1

  # Should return a torch array of shape (C, H, W) 
  def __getitem__(self, index):
    file_path = str(self.indices[index]) + ".npy"
    arr = np.load(self.dir_path + file_path, mmap_mode='r')
    if len(arr.shape) == 2:
        h,w = arr.shape
    elif len(arr.shape) != 3:
        raise Exception("Wrong number of dimensions encountered while loading data")
    data = torch.from_numpy(arr).view(-1, h, w).float()
    target = int(self.labels[self.indices[index]][0])
    
    target-=1 #expects labels to be indices of classes not values
    
    target = torch.LongTensor([target])
    if self.transform:
      data = self.transform(data)
    # For unsupervised learning, we don't need the target. 
    # return data
    # For supervised learning, we need the target. 
    return (data, target)

  def __len__(self):
    return len(self.indices)

        


