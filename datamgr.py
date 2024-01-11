import os
import sys
import torch
import numpy as np
import math
import random
import torchvision.transforms as transforms
from abc import abstractmethod
from PIL import ImageEnhance, Image

from PIL import ImageFilter, ImageOps
import torchvision.transforms.functional as TF
import utils
import timm
from timm.data.transforms_factory import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, ToTensor


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return 
        

def new_data_aug_generator(args = None):
    img_size = args.input_size
    remove_random_resized_crop = args.src
    mean, std = [0.4430, 0.3468, 0.3854], [0.1513, 0.1779, 0.1573]
    primary_tfl = []
    scale=(0.08, 1.0)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

        
    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                              Solarization(p=1.0),
                                              GaussianBlur(p=1.0)])]
   
    if args.color_jitter is not None and not args.color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))
    final_tfl = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return transforms.Compose(primary_tfl+secondary_tfl+final_tfl)
        

def build_transform(args, aug):
    resize_im = args.input_size > 32
    mean, std = [0.4430, 0.3468, 0.3854], [0.1513, 0.1779, 0.1573]
    if aug:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            mean=mean,
            std=std
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(t)


class DataAugmentationDINO:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=0.05, std=0.224),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(55, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        channels = len(image)
        
        mean = [0.052, 0.071, 0.068, 0.245, 0.111, 0.201, 0.228, 0.183, 0.114]
        std = [0.023, 0.024, 0.029, 0.058, 0.029, 0.044, 0.049, 0.041, 0.033]
        normalize = transforms.Normalize(mean=mean, std=std)
        
        seed1 = np.random.randint(2147483647)
        transfo1_imgs = []
        for i in range(channels):
            random.seed(seed1) 
            torch.manual_seed(seed1)
            transfo1_imgs.append(self.global_transfo1(image[i]))
            
        seed2 = np.random.randint(2147483647)
        transfo2_imgs = []
        for i in range(channels):
            random.seed(seed2) 
            torch.manual_seed(seed2)
            transfo2_imgs.append(self.global_transfo2(image[i]))
            
        transfo1_imgs = torch.cat(transfo1_imgs)
        transfo2_imgs = torch.cat(transfo2_imgs)
        transfo1_imgs = normalize(transfo1_imgs)
        transfo2_imgs = normalize(transfo2_imgs)
        crops.append(transfo1_imgs)
        crops.append(transfo2_imgs)
        
        for _ in range(self.local_crops_number):
            seed = np.random.randint(2147483647)
            local_transfo_imgs = []
            for j in range(channels):
                random.seed(seed) 
                torch.manual_seed(seed)
                local_transfo_imgs.append(self.local_transfo(image[j]))
                
            local_transfo_imgs = torch.cat(local_transfo_imgs)
            local_transfo_imgs = normalize(local_transfo_imgs)
            crops.append(local_transfo_imgs)
            
        return crops