from __future__ import print_function

import argparse
import csv
import os
import sys
import tifffile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import glob
import pickle

import datetime
import math
import json
import time

import timm
from timm.models import create_model
from timm.scheduler import CosineLRScheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict

from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from customloss import CustomLoss
from datamgr import *
import utils
from engine import *
from dataset import *
from models import *
from samplers import RASampler

import copy
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description= 'CBNA training script')
    parser.add_argument('--model_type'       , default='MLP', help='CNN/MLP/ViT/Fusion')
    parser.add_argument('--data_split'       , default='train', help='train/test')
    parser.add_argument('--get'       , default='predictions', help='features/predictions')
    parser.add_argument('--data_set'     , default='CBNA')
    parser.add_argument('--data_path'     , default='Data/irc_patches/')
    parser.add_argument('--file_path'     , default='datafile/')
    parser.add_argument('--output_dir', default='checkpoints/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--input_size'  , default=224, type=int, help ='Image size for training')
    parser.add_argument('--drop_path'   , type=float, default=0.05)
    parser.add_argument('--eval_crop_ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--seed'       , default=0, type=int, help='seed')

    return parser.parse_args()


def feature_extractor(model, data_loader, args):
    
    features_t = []
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(data_loader, file=sys.stdout):
            if args.model_type=="MLP":
                samples = data
                samples = samples.to(args.device, non_blocking=True)
                features = model.meta(samples)
            elif args.model_type=="CNN" or args.model_type=="ViT":
                samples = data
                samples = samples.to(args.device, non_blocking=True)
                model.base_model.reset_classifier(0)
                features = model.base_model(samples)
            elif args.model_type=="Fusion":
                samples_img, samples_meta = data
                samples_img = samples_img.to(args.device, non_blocking=True)
                samples_meta = samples_meta.to(args.device, non_blocking=True)
                
                ft_img = model.base_model(samples_img)
                ft_meta = model.meta(samples_meta)
                features = torch.cat((ft_img, ft_meta), dim=1)
                
            targets = targets.to(args.device, non_blocking=True)
            features_t.append(features.detach().cpu())
            
        features_t = torch.cat(features_t)
        torch.save(features_t, args.output_dir.joinpath('{}_{}_features.pth'.format(args.model_type, args.data_split)))
        
        return
    

def get_predictions(model, data_loader, args):
    
    predictions_t = []
    targets_t = []
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(data_loader, file=sys.stdout):
            if args.model_type=="MLP" or args.model_type=="CNN" or args.model_type=="ViT":
                samples = data
                samples = samples.to(args.device, non_blocking=True)
                outputs = model(samples)
            elif args.model_type=="Fusion":
                samples_img, samples_meta = data
                samples_img = samples_img.to(args.device, non_blocking=True)
                samples_meta = samples_meta.to(args.device, non_blocking=True)
                outputs = model(samples_img, samples_meta)
                    
            targets = targets.to(args.device, non_blocking=True)
            targets_t.append(targets.detach().cpu())
            predictions = torch.sigmoid(outputs)
            predictions_t.append(predictions.detach().cpu())
        
        targets_t = torch.cat(targets_t)
        predictions_t = torch.cat(predictions_t)
        save_dict = {'predictions': predictions_t, 'targets': targets_t}
        torch.save(save_dict, args.output_dir.joinpath('{}_{}_predictions.pth'.format(args.model_type, args.data_split)))
        
        return
    

def main(args):
    
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    #print(args)
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    meta_features = None
    if args.model_type=="MLP" or args.model_type=="Fusion":
        meta_features = ['LANDOLT_MOIST',
   'N_prct', 'pH', 'CN', 'TMeanY', 'TSeason', 'PTotY', 'PSeason', 'RTotY',
   'RSeason', 'AMPL', 'LENGTH', 'eauvive', 'clay', 'silt', 'sand', 'cv_alti']

    if args.model_type=="MLP":
        dataset_train = CBNA(args, split='train', meta_features=meta_features)
        dataset_test = CBNA(args, split='test', meta_features=meta_features)
    else: 
        dataset_train = CBNA(args, split='train', meta_features=meta_features, transform=build_transform(args, aug=False))
        dataset_test = CBNA(args, split='test', meta_features=meta_features, transform=build_transform(args, aug=False))
    
    print(f"Data loaded: there are {len(dataset_train)} train images, {len(dataset_test)} test images.")
    
    if args.model_type=="MLP" or args.model_type=="Fusion":
        args.n_meta_features = len(dataset_train.meta_features)
    
    args.train_label_cnt = dataset_train.get_label_count()
    args.test_label_cnt = dataset_test.get_label_count()
    
    if args.data_split == "train":
        sampler = torch.utils.data.SequentialSampler(dataset_train)
        data_loader = DataLoader(dataset_train, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    elif args.data_split == "test":
        sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader = DataLoader(dataset_test, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    print(f"Creating model: {args.model_type}")
    if args.model_type == 'MLP':
        model = Mlp_CBNA(args, out_dim=args.num_classes)
    elif args.model_type == 'CNN':
        model = Resnet_CBNA(args, out_dim=args.num_classes)
    elif args.model_type == 'Fusion':
        model = Fusion_CBNA(args, out_dim=args.num_classes)
    elif args.model_type == 'ViT':
        model = Vit_CBNA(args, out_dim=args.num_classes)
    model.to(args.device)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    reload_file = utils.get_best_file(args.output_dir)        
    print("best_file" , reload_file)
    checkpoint = torch.load(reload_file)
    model.load_state_dict(checkpoint['model'])
    print("model loaded")
    
    if args.get == 'features':
        print('Starting feature extraction...')
        feature_extractor(model, data_loader, args)
        print('{} features saved for {} model!'.format(args.data_split, args.model_type))
    elif args.get == 'predictions':
        print('Getting predictions and targets...')
        get_predictions(model, data_loader, args)
        print('{} predictions and targets saved for {} model!'.format(args.data_split, args.model_type))
    
if __name__ == '__main__':
    args = parse_args()
    output_dir = Path(args.output_dir).joinpath('{}/{}/'.format(args.data_set, args.model_type))
    args.output_dir = output_dir
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok = True)
            print("Directory '%s' created successfully" %args.output_dir)
        except OSError as error:
            print("Directory '%s' can not be created")

    if args.data_set == 'CBNA':
        args.classes = list(np.genfromtxt(Path(args.file_path).joinpath('classes.txt'), dtype='str'))
        args.num_classes = len(args.classes)
        print('number of labels: ', args.num_classes)
        
    main(args)