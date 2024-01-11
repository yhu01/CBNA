import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
import tifffile
import pandas as pd
import torch.nn as nn
from pathlib import Path
from timm.data.parsers.parser import Parser
from PIL import ImageEnhance, Image
import matplotlib.pyplot as plt
from pathlib import Path
from timm.data import create_transform
from torchvision.utils import make_grid
from datamgr import *
import rasterio as rio
#from osgeo import gdal
import os


class Alps_meta(Dataset):
    def __init__(self, args, meta_features):
        #targets = list of species CD_REF , column names
        self.data_path = Path(args.data_path)
        self.data_df = pd.read_csv(args.file_path+'alps_env_data.csv', sep=',', low_memory=False)
        self.meta_features = meta_features
        self.meta_stats = torch.load(args.file_path+'stats_meta.pth')
        

    def __len__(self):
        return len(self.data_df)
    
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        coord = torch.tensor(row[['X', 'Y']]).float()
        data = torch.tensor(row[self.meta_features]).float()
        data = (data - self.meta_stats['mean'][:-1]) / self.meta_stats['std'][:-1]
        
        return coord, data


class CBNA(Dataset):
    def __init__(self, args, split='train', meta_features=None, transform=None):
        #targets = list of species CD_REF , column names
        self.data_path = Path(args.data_path)
        if split == 'train':
            self.data_df = pd.read_csv(args.file_path+'{}_w_covariates_final.csv'.format(split), sep=',', low_memory=False, dtype={'label': str})
        else:
            self.data_df = pd.read_csv(args.file_path+'{}_w_covariates.csv'.format(split), sep=',', low_memory=False, dtype={'label': str})
        self.classes = args.classes
        self.transform = transform
        self.model_type = args.model_type
        
        if meta_features is not None:
            self.meta_features = meta_features
            self.meta_stats = torch.load(args.file_path+'stats_meta.pth')
        
        
    def __len__(self):
        return len(self.data_df)
    
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        img_id, target = row['id_img'], row['labelset']
        
        if self.model_type == "CNN" or self.model_type == "ViT" or self.model_type == "Fusion": 
            data_path = self.data_path.joinpath('plot%s.tif'%(img_id))
            image = tifffile.imread(data_path)
            image = Image.fromarray(image.astype(np.uint8).transpose(1,2,0))

            if self.transform:
                image = self.transform(image)

            if self.model_type == "Fusion":
                data_meta = torch.tensor(self.data_df.iloc[idx][self.meta_features]).float()
                #data_meta = (data_meta - self.meta_stats['mean'].squeeze(0)) / self.meta_stats['std'].squeeze(0)
                data = (image, data_meta)
            else:
                data = image
                
        elif self.model_type == "MLP":
            data = torch.tensor(self.data_df.iloc[idx][self.meta_features]).float()
            data = (data - self.meta_stats['mean']) / self.meta_stats['std']
        
        return data, self.encode(target)

    
    def get_cdref(self, idx):
        row = self.data_df.iloc[idx]
        cd_ref = row['labelset']
        
        #print('cd_ref for the index {}: '.format(idx), list(map(float, cd_ref.split(','))))
        return list(map(float, cd_ref.split(',')))
    
    
    def get_species(self, idx, cdref2species):
        cdref = self.get_cdref(idx)
        species = []
        for ref in cdref:
            temp = cdref2species.loc[cdref2species.cd_ref==ref]
            species.append(temp.nom_reconnu.values[0])
        return species
    
    
    def encode(self, target):
        num_cls = len(self.classes)
        vec = torch.zeros(num_cls)
        label_list = target.split(',')
        for l in label_list:
            idx = self.classes.index(l)
            vec[idx] = 1
        return vec


    def decode(self, output, threshold=0.5):
        idx = (output >= threshold).nonzero().flatten()
        cdref = []
        for i in idx.tolist():
            cdref.append(self.classes[i])
        return ','.join(cdref)
    
    
    def get_label_count(self):
        num_cls = len(self.classes)
        init_values = [0] * num_cls
        label_cnt = dict(zip(self.classes, init_values))
        
        labelset_arrays = self.data_df.labelset.values
        for i in range(len(labelset_arrays)):
            labelset = labelset_arrays[i].split(',')
            for label in labelset:
                label_cnt[label] += 1
                
        return label_cnt
    
    
    def show_sample(self, idx):
        row = self.data_df.iloc[idx]
        img_id, target = row['id_img'], row['labelset']
        
        data_path = self.data_path.joinpath('plot%s.tif'%(img_id))
        image = tifffile.imread(data_path)
        image = Image.fromarray(image.astype(np.uint8).transpose(1,2,0))
        
        plt.imshow(image)
        print(target)
        return image
        
        
    def show_batch(self, data_loader, n):
        j = 0
        for images, targets in data_loader:
            j = j + 1
            if j == n:
                # print(images.var(dim=(1,2,3)))
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
                break
            else:
                continue

                
def build_dataset(args, is_train=True):
    
    if args.data_set == 'CBNA':
        
        meta_features = None
        if args.model_type=="MLP" or args.model_type=="Fusion":
            meta_features = ['LANDOLT_MOIST',
       'N_prct', 'pH', 'CN', 'TMeanY', 'TSeason', 'PTotY', 'PSeason', 'RTotY',
       'RSeason', 'AMPL', 'LENGTH', 'eauvive', 'clay', 'silt', 'sand', 'cv_alti']
        
        if is_train:
            if args.model_type=="MLP":
                dataset_train = CBNA(args, split='train', meta_features=meta_features)
                dataset_val = CBNA(args, split='val', meta_features=meta_features)
            else: 
                dataset_train = CBNA(args, split='train', meta_features=meta_features, transform=build_transform(args, aug=True))
                dataset_val = CBNA(args, split='val', meta_features=meta_features, transform=build_transform(args, aug=False))
            
            return dataset_train, dataset_val
        
        else:
            if args.model_type=="MLP":
                dataset_test = CBNA(args, split='test', meta_features=meta_features)
            else:
                dataset_test = CBNA(args, split='test', meta_features=meta_features, transform=build_transform(args, aug=False))
            
            return dataset_test



