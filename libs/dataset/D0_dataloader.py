import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import deepdish as dd
from PIL import Image
import csv
import numpy as np
from time import sleep
# import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageChops
# Image.LOAD_TRUNCATED_IMAGES = True

# class MySampler(torch.utils.data.Sampler):
#     def __init__(self, end_idx, seq_length):
#         indices = []
#         for i in range(len(end_idx) - 1):
#             start = end_idx[i]
#             end = end_idx[i + 1] - seq_length
#             indices.append(torch.arange(start, end))
#         indices = torch.cat(indices)
#         self.indices = indices
#
#     def __iter__(self):
#         indices = self.indices[torch.randperm(len(self.indices))]
#         return iter(indices.tolist())
#
#     def __len__(self):
#         return len(self.indices)


class MyDataset(Dataset):
    def __init__(self, data_paths,transform):
        self.data_paths = data_paths
        self.transform = transform

    def __getitem__(self, index):


        image_rgb_before=Image.open(self.data_paths+str(index)+'/kinectA_rgb_before.jpg')
        image_rgb_during = Image.open(self.data_paths  + str(index) + '/kinectA_rgb_during.jpg')
        image_gelsight0_before = Image.open(self.data_paths + str(index) + '/gelsightA_before.jpg')
        image_gelsight0_during = Image.open(self.data_paths + str(index) + '/gelsightA_during.jpg')
        image_gelsight1_before = Image.open(self.data_paths  + str(index) + '/gelsightB_before.jpg')
        image_gelsight1_during = Image.open(self.data_paths + str(index) + '/gelsightB_during.jpg')
        # rgb=ImageChops.subtract(image_rgb_during,image_rgb_before)
        rgb = image_rgb_during
        gel= Image.new('RGB',(1280,960*2))
        gel.paste(ImageChops.subtract(image_gelsight0_during,image_gelsight0_before),(0,0,1280,960))
        gel.paste(ImageChops.subtract(image_gelsight1_during ,image_gelsight1_before), (0, 960, 1280, 1920))
        # gel.paste(image_gelsight0_during, (0, 0, 1280, 960))
        # gel.paste(image_gelsight1_during, (0, 960, 1280, 1920))
        if np.load(self.data_paths + str(index) + '/is_gripping.npy'):
            label =torch.tensor(1, dtype=torch.long)
        else:
            label =torch.tensor(0, dtype=torch.long)
        if self.transform:
            rgb = self.transform(rgb)
            gel = self.transform(gel)
        return rgb,gel,label

    def __len__(self):
        return len(os.listdir(self.data_paths))



