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
import rasterio as rio
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
    parser.add_argument('--model_type'       , default='MLP', help='CNN/MLP')
    parser.add_argument('--data_path'     , default='Data/')
    parser.add_argument('--file_path'     , default='datafile/')
    parser.add_argument('--data_set'     , default='CBNA')
    parser.add_argument('--output_dir', default='checkpoints/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu', help='device to use for training / testing')
    parser.add_argument('--input_size'  , default=224, type=int, help ='Image size for training')
    parser.add_argument('--drop_path'   , type=float, default=0.05)
    parser.add_argument('--eval_crop_ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--seed'       , default=0, type=int, help='seed')
    parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('--batch_size', default=636, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--start_batch_idx', default=0, type=int)

    return parser.parse_args()
 

def main(args):
    
    args.checkout_path = Path('checkpoints/alps_{}/'.format(args.model_type))
    meta_features = None
    if not os.path.isdir(args.checkout_path):
        try:
            os.makedirs(args.checkout_path, exist_ok = True)
            print("Directory '%s' created successfully" %args.checkout_path)
        except OSError as error:
            print("Directory '%s' can not be created")
            
    if args.model_type == 'MLP':
        meta_features = ['LANDOLT_MOIST',
           'N_prct', 'pH', 'CN', 'TMeanY', 'TSeason', 'PTotY', 'PSeason', 'RTotY',
           'RSeason', 'AMPL', 'LENGTH', 'eauvive', 'clay', 'silt', 'sand']
        args.n_meta_features = len(meta_features)
        
        dataset_alps = Alps_meta(args, meta_features=meta_features)
        print(f"Data loaded: there are {len(dataset_alps)} alps images.")
        sampler_alps = torch.utils.data.SequentialSampler(dataset_alps)
        alps_loader = DataLoader(dataset_alps, sampler=sampler_alps, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        args.end_batch_idx = len(dataset_alps.data_df) // args.batch_size 
        print('end_batch_idx: ', args.end_batch_idx)
    else:
        args.dataframe = pd.read_csv(args.file_path+'alps_env_data.csv', sep=',', low_memory=False)
        print(f"Data loaded: there are {len(args.dataframe)} alps images.")
        args.alps_irc_mosaic = rio.open(args.data_path+'alps_mosaic.vrt')
        print('mosaic map loaded')
        args.end_batch_idx = len(args.dataframe) // args.batch_size 
        print('end_batch_idx: ', args.end_batch_idx)
    
    print(f"Creating model: {args.model_type}")
    if args.model_type == 'MLP':
        model = Mlp_CBNA(args, out_dim=args.num_classes)
    elif args.model_type == 'CNN':
        model = Resnet_CBNA(args, out_dim=args.num_classes)
    model.to(args.device)
    
    reload_file = utils.get_best_file(args.output_dir)        
    print("best_file" , reload_file)

    checkpoint = torch.load(reload_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    print("model loaded")
    
    if args.model_type == 'MLP':
        extract_features_MLP(model, alps_loader, args)
    else:
        extract_features_CNN(model, args.dataframe, args.alps_irc_mosaic, args)
    
    return


def extract_features_MLP(model, alps_loader, args):
    batch_idx = args.start_batch_idx
    model.eval()
    
    print('start batch idx: ', batch_idx)
    print('end batch idx: ', args.end_batch_idx)
    print('Starting extraction!')
    
    with torch.no_grad():
        for coord, data in tqdm(alps_loader, file=sys.stdout):
            if os.path.exists(args.checkout_path.joinpath('{}d.pth'.format(batch_idx))):
                print('Already extracted')
                continue
            
            samples_meta = data
            samples_meta = samples_meta.to(args.device, non_blocking=True)
        
            features = model.meta(samples_meta)
            predictions = torch.sigmoid(model(samples_meta))

            save_dict = {'batch_idx': batch_idx, '{}_coord'.format(args.model_type): coord.detach().cpu(), '{}_features'.format(args.model_type):features.detach().cpu(), '{}_predictions'.format(args.model_type): predictions.detach().cpu()}
            torch.save(save_dict, args.checkout_path.joinpath('{}.pth'.format(batch_idx)))
            batch_idx += 1
        print('Features saved!')
            
    
def extract_features_CNN(model, dataframe, alps_irc_mosaic, args):
    
    invalid_patches = []
    transform = build_transform(args, aug=False)
    N = 512

    start_batch_idx = args.start_batch_idx
    end_batch_idx = args.end_batch_idx

    if args.resume:
        invalid_patches = utils.load_pickle(args.checkout_path.joinpath('invalid_patches.pkl'))['invalid_patches']
        print('invalid_patches: ', invalid_patches)
        resume_file = utils.get_resume_file(args.checkout_path)
        start_batch_idx = torch.load(resume_file)['batch_idx'] + 1

    else:
        stats1 = {'invalid_patches': invalid_patches}
        print(stats1)
        utils.save_pickle(args.checkout_path.joinpath('invalid_patches.pkl'), stats1)

    print('start batch idx: ', start_batch_idx)
    print('end batch idx: ', end_batch_idx)
    print('Starting extraction!')
    
    model.eval()
    for batch_idx in tqdm(range(start_batch_idx, end_batch_idx), file=sys.stdout):

        print('Starting extraction for batch: ', batch_idx)

        if os.path.exists(args.checkout_path.joinpath('{}.pth'.format(batch_idx))):
            print('Already extracted')
            continue

        idx_of_batch = batch_idx * args.batch_size

        irc_patches = []
        irc_features = []
        coord = []

        for idx in tqdm(range(idx_of_batch, idx_of_batch+args.batch_size), file=sys.stdout):

            #if idx % 50 == 0 or idx == idx_of_batch+args.batch_size-1:
            #    print('batch [{}][{}/{}]'.format(batch_idx, idx, idx_of_batch+args.batch_size))

            irc_patch = None
            row = dataframe.iloc[idx]
            (lon, lat) = row[['X','Y']].values

            R = N // 2
            py, px = alps_irc_mosaic.index(lon, lat)
            ## get window bounds
            wind = rio.windows.Window(max(0,px - R), max(0,py - R), N, N)
            ## extract window values
            try:
                irc_patch = alps_irc_mosaic.read(window=wind).astype(np.uint8)
            except:
                print("An exception occurred")

            if irc_patch is None or irc_patch.shape != (3, N, N):
                invalid_patches.append(idx)
                print('Invalid patch idx: {}'.format(idx))
                stats1 = {'invalid_patches': invalid_patches}
                print(stats1)
                utils.save_pickle(args.checkout_path.joinpath('invalid_patches.pkl'), stats1)
                continue

            image = Image.fromarray(irc_patch.transpose(1,2,0))
            irc_patches.append(transform(image).to(args.device, non_blocking=True).unsqueeze(0))
            coord.append(torch.tensor(row[['X','Y']].values).float().unsqueeze(0))

        irc_patches = torch.cat(irc_patches)
        coord = torch.cat(coord)
        
        with torch.no_grad():
            irc_patches = irc_patches.to(args.device, non_blocking=True)
            irc_feature_maps = model.base_model.forward_features(irc_patches)
            irc_features = model.base_model.global_pool(irc_feature_maps).detach().cpu()
            predictions = torch.sigmoid(model(irc_patches)).detach().cpu()

        save_dict = {'batch_idx': batch_idx, '{}_coord'.format(args.model_type): coord, '{}_features'.format(args.model_type):irc_features, '{}_predictions'.format(args.model_type): predictions}
        torch.save(save_dict, args.checkout_path.joinpath('{}.pth'.format(batch_idx)))
        print('Features saved!')


if __name__ == '__main__':
    args = parse_args()
    if args.model_type == 'MLP': 
        output_dir = Path(args.output_dir).joinpath('{}/{}/wo_alt/'.format(args.data_set, args.model_type))
    else:
        output_dir = Path(args.output_dir).joinpath('{}/{}/'.format(args.data_set, args.model_type))
    args.output_dir = output_dir
    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok = True)
            print("Directory '%s' created successfully" %args.output_dir)
        except OSError as error:
            print("Directory '%s' can not be created")

    args.classes = list(np.genfromtxt(Path(args.file_path).joinpath('classes.txt'), dtype='str'))
    args.num_classes = len(args.classes)
    print('number of labels: ', args.num_classes)
        
    main(args)
