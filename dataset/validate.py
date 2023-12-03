from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
import torch
from collections import OrderedDict

class ValidateDataset(Dataset):
    def __init__(self, rb_size, transform, sampling):
        #self.rb_path = rb_path
        self.transform = transform
        self.rb_size = rb_size
        self.sampling = sampling

        self.data = list()
        self.targets = list()
        self.filenames = list()

        self.offset = dict()
        self.len_per_cls = dict()
    
    def __len__(self):
        #return self.filled_counter
        assert len(self.data) == len(self.targets) == len(self.filenames)

        return len(self.data)

    def __getitem__(self, idx):
        vec = self.data[idx]
        if self.transform is not None:
            vec = self.transform(vec)
        label = self.targets[idx]
        filename = self.filenames[idx]
        return idx, vec, label