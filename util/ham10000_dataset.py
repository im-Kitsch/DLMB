import os

from PIL import Image

import torchvision
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class HAM10000Dataset(Dataset):
    def __init__(self, img_root, csv_file, transform=None, report=True):
        super(HAM10000Dataset, self).__init__()
        self.img_root = os.path.join(img_root, 'img/')
        self.csv_file = pd.read_csv(csv_file, usecols=['image_id', 'dx'])
        self.transform = transform
        if report is True:
            print('statistics of HAM10000')
            print(self.csv_file.loc[:, 'dx'].value_counts())
        return

    def __len__(self):
        return self.csv_file.shape[0]

    def __getitem__(self, idx):
        img_file, img_lbl = self.csv_file.loc[idx, 'image_id'], self.csv_file.loc[idx, 'dx']
        img_path = os.path.join(self.img_root, img_file+'.jpg')
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_lbl
