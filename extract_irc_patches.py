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

import rasterio as rio
import utils
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid

import copy
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description= 'CBNA IRC patch extraction script')
    
    parser.add_argument('--start_idx'  , default=66180, type=int, help ='Starting epoch')
    parser.add_argument('--end_idx'  , default=162293, type=int, help ='Stopping epoch')
    parser.add_argument('--resume'   , type=utils.bool_flag, default=True)

    return parser.parse_args()


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

        
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    

def extract_patches(res, alps_irc_mosaic, args, N=512):
    lis = []
    invalid_img = []
    start_idx = args.start_idx
    
    if args.resume:
        invalid_img = load_pickle('./invalid_patch.pkl')['invalid_img']
        start_idx = load_pickle('./restore_idx.pkl')['restore_idx'] 
    
    print('start_idx: ', start_idx)
    end_idx = args.end_idx
    print('end_idx: ', end_idx)
    for idx in range(start_idx, end_idx):
        irc_patch = None
        row = res.iloc[idx]
        (lon, lat) = row[['x_l93','y_l93']].values

        R = N // 2
        py, px = alps_irc_mosaic.index(lon,lat)
        ## get window bounds
        wind = rio.windows.Window(max(0,px - R), max(0,py - R), N, N)
        ## extract window values
        try:
            irc_patch = alps_irc_mosaic.read(window=wind).astype(np.uint8)
        except:
            ValueError('Error')

        id_img = row['id_img']
        print('{}: {}'.format(idx, id_img))
        if os.path.exists('Data/irc_patches/plot%d.tif'%id_img):
            stats2 = {'restore_idx': idx}
            save_pickle('./restore_idx.pkl', stats2)
            print('Already processed')
            continue

        if irc_patch is None:
            invalid_img.append(idx)
            print('Invalid image: {}'.format(id_img))
            stats1 = {'invalid_img': invalid_img}
            save_pickle('./invalid_patch.pkl', stats1)
            continue
            
        stats2 = {'restore_idx': idx}
        save_pickle('./restore_idx.pkl', stats2)
        tifffile.imwrite('Data/irc_patches/plot%d.tif'%id_img, irc_patch, photometric='minisblack', compression='zlib')
        
    
def main(args):
    extract_patches(args.res, args.alps_irc_mosaic, args, N=512)
    
    
if __name__ == '__main__':
    args = parse_args()
    args.res = pd.read_csv('filelist/cbna_norepeat.csv',sep=',',low_memory=False)
    print('total length: ', len(args.res))
    args.alps_irc_mosaic = rio.open('Data/alps_mosaic.vrt')
    main(args)