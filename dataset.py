import os

import numpy as np
import PIL.Image
import torch
from torch.utils import data
import pdb
import random
import cv2


class MyData(data.Dataset):
    """
    load images for testing
    root: director/to/images/
            structure:
            - root
                - images (images here)
                - masks (ground truth)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, transform=True, hflip=False, vflip=False, crop=False):
        super(MyData, self).__init__()
        self.root = root
        self.is_transform = transform
        self.is_hflip = hflip
        self.is_vflip = vflip
        self.is_crop = crop
        img_root = os.path.join(self.root, 'images')
        gt_root = os.path.join(self.root, 'masks')
        file_names = os.listdir(gt_root)
        self.img_names = []
        self.map_names = []
        self.gt_names = []
        self.names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.png'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.gt_names.append(
                os.path.join(gt_root, name[:-4] + '.png')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]

        gt_file = self.gt_names[index]
        gt = PIL.Image.open(gt_file)
        gt = np.array(gt, dtype=np.int32)
        gt[gt != 0] = 1
        if self.is_crop:
            H = int(0.9 * img.shape[0])
            W = int(0.9 * img.shape[1])
            H_offset = random.choice(range(img.shape[0] - H))
            W_offset = random.choice(range(img.shape[1] - W))
            H_slice = slice(H_offset, H_offset + H)
            W_slice = slice(W_offset, W_offset + W)
            img = img[H_slice, W_slice, :]
            gt = gt[H_slice, W_slice]
        if self.is_hflip and random.randint(0, 1):
            img = img[:, ::-1, :]
            gt = gt[:, ::-1]
        if self.is_vflip and random.randint(0, 1):
            img = img[::-1, :, :]
            gt = gt[::-1, :]
        img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        gts = [cv2.resize(gt, dsize=(sz, sz), interpolation=cv2.INTER_NEAREST) for sz in [256, 128, 64, 32, 16]]

        if self.is_transform:
            img, gts = self.transform(img, gts)
            return img, gts
        else:
            return img, gts

    def transform(self, img, gts):
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        gts = [torch.from_numpy(gt) for gt in gts]
        return img, gts


class MyTestData(data.Dataset):
    """
    load images for testing
    root: director/to/images/
            structure:
            - root
                - images (images here)
                - masks (ground truth)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, root, transform=True):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'images')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img_size = img.size
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)
        if self._transform:
            img = self.transform(img)
            return img, self.names[index], img_size
        else:
            return img, self.names[index], img_size

    def transform(self, img):
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img