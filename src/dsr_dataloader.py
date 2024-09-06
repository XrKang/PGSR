# -*- coding: utf-8 -*-

from __future__ import division

import torch
import torch.nn as nn
import logging
from scipy.io import loadmat,savemat

from PIL import Image, ImageOps
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from skimage.io import imread, imsave

from torchvision.transforms import Compose
from torchvision import transforms
import random
import numpy as np
import h5py
# import cv2
from skimage.io import imread, imsave


def get_patch(depth, rgb, patch_size):
    ih, iw = depth.shape
    tp = patch_size  # target_patch_size

    tx = random.randrange(0, iw - tp + 1)
    ty = random.randrange(0, ih - tp + 1)

    depth_patch  = depth[ty:ty + tp, tx: tx + tp]
    rgb_patch    = rgb[:, ty:ty + tp, tx: tx + tp]

    return depth_patch, rgb_patch


def augment(depth, rgb):

    if random.random() < 0.5:
        # Random vertical Flip
        rgb = rgb[:, :, ::-1].copy()
        depth = depth[:, ::-1].copy()

    if random.random() < 0.5:
        # Random horizontal Flip
        rgb = rgb[:, ::-1, :].copy()
        depth = depth[::-1, :].copy()

    # if random.random() < 0.5:
    #     # Random rotation
    #     rgb = np.rot90(rgb.copy(), axes=(1, 2))
    #     depth = np.rot90(depth.copy(), axes=(0, 1))

    return depth, rgb


def get_patch_triple(rgb, depth_lr, depth_hr, patch_size, scale):
    assert patch_size % scale == 0

    ih, iw = depth_lr.shape

    ip = patch_size // scale  # input_patch_size
    tp = patch_size  # target_patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    depth_lr_patch = depth_hr[ix: ix + ip, iy: iy + ip]
    depth_hr_patch = depth_hr[ty:ty + tp, tx: tx + tp]
    rgb_patch      = rgb[:, ty:ty + tp, tx: tx + tp]

    return rgb_patch, depth_lr_patch, depth_hr_patch



def augment_triple(rgb, depth_lr, depth_hr):

    if random.random() < 0.5:
        # Random vertical Flip
        rgb = rgb[:, :, ::-1].copy()
        depth_lr = depth_lr[:, ::-1].copy()
        depth_hr = depth_hr[:, ::-1].copy()

    if random.random() < 0.5:
        # Random horizontal Flip
        rgb = rgb[:, ::-1, :].copy()
        depth_lr = depth_lr[::-1, :].copy()
        depth_hr = depth_hr[::-1, :].copy()

    # if random.random() < 0.5:
    #     # Random rotation
    #     rgb = np.rot90(rgb.copy(), axes=(1, 2))
    #     depth_lr = np.rot90(depth_lr.copy(), axes=(1, 2))
    #     depth_hr = np.rot90(depth_hr.copy(), axes=(1, 2))

    return rgb, depth_lr, depth_hr

class Train_NYU(Dataset):
    def __init__(self, args):
        super(Train_NYU, self).__init__()

        self.depth_path = args.depth_path
        self.depth_names = os.listdir(self.depth_path)

        self.rgb_path = args.rgb_path

        self.scale = args.scale
        self.patch_size = args.patch_size # HR patch size
        self.data_augmentation = args.augmentation

    def __len__(self):
        return len(self.depth_names)

    def __getitem__(self, index):
        depth_name = os.path.join(self.depth_path, self.depth_names[index])
        depth = np.load(depth_name) # [H, W]

        rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-4]+".jpg")
        rgb = imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0        # [H, W, 3]
        rgb = np.transpose(rgb, [2, 0, 1])  # [3, xx, xx]

        # crop patch
        depth_patch, rgb_patch = get_patch(depth, rgb, self.patch_size)

        # data augmentation
        if self.data_augmentation:
            depth_patch, rgb_patch = augment(depth_patch, rgb_patch)

        # normalize
        depth_min = depth_patch.min()
        depth_max = depth_patch.max()
        depth_patch = (depth_patch - depth_min) / (depth_max - depth_min)
        rgb_patch = (rgb_patch - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
            [0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # depth resize
        h, w = depth_patch.shape[:2]
        depth_patch_lr = np.array(
            Image.fromarray(depth_patch).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        # to tensor
        depth_patch = torch.from_numpy(depth_patch.astype(np.float32)).unsqueeze(dim=0).contiguous()    # [1, H, W]
        depth_patch_lr = torch.from_numpy(depth_patch_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        rgb_patch = torch.from_numpy(rgb_patch.astype(np.float32)).contiguous() # [3, H, W]

        return depth_patch_lr, rgb_patch, depth_patch


class ValidLoader(Dataset):
    def __init__(self, args):
        super(ValidLoader, self).__init__()
        self.dataset_name = args.dataset_name

        self.depth_path = args.depth_path_valid
        self.depth_names = os.listdir(self.depth_path)

        self.rgb_path = args.rgb_path_valid

        self.scale = args.scale

    def __len__(self):
        return len(self.depth_names)

    def __getitem__(self, index):
        depth_name = os.path.join(self.depth_path, self.depth_names[index])
        if "NYU" in self.dataset_name:
            depth = np.load(depth_name)  # [H, W]
        elif "Middlebury" in self.dataset_name:
            depth = imread(depth_name).astype(np.float32)  # [H, W]
        elif "Lu" in self.dataset_name:
            depth = imread(depth_name).astype(np.float32)  # [H, W]
        elif "RGBDD" in self.dataset_name:
            depth = imread(depth_name).astype(np.float32) / 1000  # [h,w] 0~3000+(mm)


        if "NYU" in self.dataset_name:
            rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-4] + ".jpg")
        elif "Middlebury" in self.dataset_name:
            rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9] + "color.png")  # depth.png
        elif "Lu" in self.dataset_name:
            rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9] + "color.png")  # depth.png
        elif "RGBDD" in self.dataset_name:
            rgb_name = os.path.join(self.rgb_path, self.depth_names[index][:-9] + "RGB.jpg")

        rgb = imread(rgb_name)
        rgb = rgb.astype(np.float32) / 255.0  # [H, W, 3]
        rgb = np.transpose(rgb, [2, 0, 1])  # [3, H, W]


        # normalize
        depth_min = depth.min()
        depth_max = depth.max()
        depth = (depth - depth_min) / (depth_max - depth_min)
        rgb = (rgb - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array(
            [0.229, 0.224, 0.225]).reshape(3, 1, 1)

        h, w = depth.shape[:2]
        depth = depth[:h//16*16, :w//16*16]
        rgb = rgb[:, :h//16*16, :w//16*16]

        # depth resize
        h, w = depth.shape[:2]
        depth_lr = np.array(
            Image.fromarray(depth).resize((w // self.scale, h // self.scale), Image.BICUBIC))

        # to tensor
        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(dim=0).contiguous()    # [1, H, W]
        depth_lr = torch.from_numpy(depth_lr.astype(np.float32)).unsqueeze(dim=0).contiguous()  # [1, H, W]
        rgb = torch.from_numpy(rgb.astype(np.float32)).contiguous() # [3, H, W]

        return depth_lr, rgb, depth, depth_min, depth_max, self.depth_names[index][:-4]


