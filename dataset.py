import os
import torch.utils.data as data
from PIL import Image
import pandas as pd


class DDSM_dataset(data.Dataset):
    def __init__(self, root, view_laterality, split, transform):
        self.root = root
        self.view_laterality = view_laterality
        self.split = split
        self.transform = transform

        def process_line(line):
            mg_examination_id, lcc_img_id, rcc_img_id, lmlo_img_id, rmlo_img_id, breast_density = line.strip().split(',')
            return mg_examination_id, lcc_img_id, rcc_img_id, lmlo_img_id, rmlo_img_id, breast_density

        with open(os.path.join(self.root, 'data_frame/{}_density.txt'.format(self.split)), 'r') as f:
            self.data = list(map(process_line, f.readlines()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series_id, lcc_img_id, rcc_img_id, lmlo_img_id, rmlo_img_id, density = self.data[idx]

        if self.view_laterality == 'lcc':
            image = Image.open(os.path.join(self.root, 'images', series_id, lcc_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'lmlo':
            image = Image.open(os.path.join(self.root, 'images', series_id, lmlo_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'rcc':
            image = Image.open(os.path.join(self.root, 'images', series_id, rcc_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'rmlo':
            image = Image.open(os.path.join(self.root, 'images', series_id, rmlo_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'left':
            cc_img = Image.open(os.path.join(self.root, 'images', series_id, lcc_img_id + '.jpg')).convert('RGB')
            mlo_img = Image.open(os.path.join(self.root, 'images', series_id, lmlo_img_id + '.jpg')).convert('RGB')
        elif self.view_laterality == 'right':
            cc_img = Image.open(os.path.join(self.root, 'images', series_id, rcc_img_id + '.jpg')).convert('RGB')
            mlo_img = Image.open(os.path.join(self.root, 'images', series_id, rmlo_img_id + '.jpg')).convert('RGB')
        else:
            lcc_img = Image.open(os.path.join(self.root, 'images', series_id, lcc_img_id + '.jpg')).convert('RGB')
            lmlo_img = Image.open(os.path.join(self.root, 'images', series_id, lmlo_img_id + '.jpg')).convert('RGB')
            rcc_img = Image.open(os.path.join(self.root, 'images', series_id, rcc_img_id + '.jpg')).convert('RGB')
            rmlo_img = Image.open(os.path.join(self.root, 'images', series_id, rmlo_img_id + '.jpg')).convert('RGB')

        if self.view_laterality == 'lcc' or self.view_laterality == 'lmlo' or self.view_laterality == 'rcc' or self.view_laterality == 'rmlo':
            image = self.transform(image)
            return image, density

        if self.view_laterality == 'left' or self.view_laterality == 'right':
            cc_img = self.transform(cc_img)
            mlo_img = self.transform(mlo_img)
            return cc_img, mlo_img, density

        else:
            lcc_img = self.transform(lcc_img)
            lmlo_img = self.transform(lmlo_img)
            rcc_img = self.transform(rcc_img)
            rmlo_img = self.transform(rmlo_img)
            return lcc_img, lmlo_img, rcc_img, rmlo_img, density

