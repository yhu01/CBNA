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
    
    parser.add_argument('--data_set'     , default='CBNA')
    parser.add_argument('--data_path'     , default='Data/irc_patches/')
    parser.add_argument('--file_path'     , default='datafile/')
    parser.add_argument('--output_dir', default='checkpoints/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    
    parser.add_argument('--epochs'  , default=100, type=int, help ='Stopping epoch')
    parser.add_argument('--input_size'  , default=224, type=int, help ='Image size for training')
    parser.add_argument('--eval_crop_ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m7-mstd0.5', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \"(default: None)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--clip_grad', type=float, default=0., help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    
    parser.add_argument('--use_fp16'    , type=utils.bool_flag, default=False)
    parser.add_argument('--repeated_aug'    , type=utils.bool_flag, default=True)
    parser.add_argument('--ThreeAugment'    , type=utils.bool_flag, default=False)
    parser.add_argument('--lr'          , default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay'          , default=5e-2, type=float, help='weight decay')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--val_batch_size'  , default=128, type=int, help='batch size for validation')
    parser.add_argument('--loss_fn'     , default='focal', help='bce/focal/dice')
    parser.add_argument('--weighted'    , type=utils.bool_flag, default=True)
    parser.add_argument('--drop_path'   , type=float, default=0.05)
    parser.add_argument('--src', action='store_true') #simple random crop
    parser.add_argument('--eval'        , action='store_true')
    parser.add_argument('--seed'       , default=0, type=int, help='seed')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    
    parser.add_argument('--thres_method'   , default='global', help='global/adaptive')
    parser.add_argument('--threshold'      , default=0.5, type=float, help='compute TSS, 0.45 for CNN & Fusion, 0.5 for MLP')

    return parser.parse_args()

    
def main(args):
    
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    #print(args)
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    dataset_train, dataset_val = build_dataset(args, is_train=True)
    args.train_label_cnt = dataset_train.get_label_count()
    print(f"Data loaded: there are {len(dataset_train)} train images, {len(dataset_val)} val images.")
    
    if args.model_type=="Fusion" or args.model_type=="MLP":
        args.n_meta_features = len(dataset_train.meta_features)
    
    if args.eval:
        dataset_test = build_dataset(args, is_train=False)
        args.test_label_cnt = dataset_test.get_label_count()
        print(f"Data loaded: there are {len(dataset_test)} test images.")
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        test_loader = DataLoader(dataset_test, sampler=sampler_test, batch_size=args.val_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        
    else:
        if args.repeated_aug:
            sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_train = DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        train_loader = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        if args.ThreeAugment:
            train_loader.dataset.transform = new_data_aug_generator(args)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        val_loader = DataLoader(dataset_val, sampler=sampler_val, batch_size=args.val_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
        
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
    
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model = DDP(model)
    model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    criterion = CustomLoss(args.loss_fn, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineLRScheduler(optimizer, t_initial = args.epochs, lr_min=1e-5, warmup_t = 5, warmup_lr_init = 1e-4)
        
    # ============ optionally resume training ... ============
    start_epoch = 0
    best_tss = 0.
    if args.resume:
        reload_file = utils.get_resume_file(args.output_dir)        
        print("resume_file" , reload_file)

        checkpoint = torch.load(reload_file, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("model loaded")

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("optimizer loaded")
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("scheduler loaded")
            start_epoch = checkpoint['epoch'] + 1
            print("restored epoch is" , checkpoint['epoch'])
            
            if 'best_perf' in checkpoint:
                best_tss = checkpoint['best_perf']
                print("current best performance is" , best_tss)
            
            if 'fp16_scaler' in checkpoint:
                fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])

        lr_scheduler.step(start_epoch)
        
    if args.eval:
        reload_file = utils.get_best_file(args.output_dir)        
        print("best_file" , reload_file)

        checkpoint = torch.load(reload_file, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("model loaded")

        test_stats = evaluate(model, test_loader, criterion, args)
        print(f"True Skill Score of the network on the {len(dataset_val)} test images: macro: {test_stats['macro_tss']:.4f}%, micro: {test_stats['micro_tss']:.4f}%, weighted: {test_stats['weighted_tss']:.4f}%")
        return
    
    print("Starting/Resuming CBNA training !")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # Training Phase 
        train_loader.sampler.set_epoch(epoch)
        
        # ============ training one epoch of CBNA ... ============
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, epoch, fp16_scaler, args)
        lr_scheduler.step(epoch)
        
        # ============ evaluating ... ==============
        val_stats = evaluate(model, val_loader, criterion, args)
        print(f"True Skill Score of the network on the {len(dataset_val)} val images: {val_stats['macro_tss']:.4f}%")
            
        save_dict = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'current_perf': val_stats['macro_tss'],
            'best_perf': best_tss,
            'args': args,
        }
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        if val_stats['macro_tss'] > best_tss:
            print('Best performance obtained!')
            best_tss = val_stats['macro_tss']
            save_dict['best_perf'] = best_tss
            best_output_dir = args.output_dir.joinpath('best.pth')
            utils.save_on_master(save_dict, best_output_dir)
        
        # ============ writing logs ... ============
        output_dir = args.output_dir.joinpath('{:d}.pth'.format(epoch))
        utils.save_on_master(save_dict, output_dir)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        if utils.is_main_process():
            with args.output_dir.joinpath("log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
def train_one_epoch(model, train_loader, criterion, optimizer, epoch, fp16_scaler, args):
    
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for data, targets in metric_logger.log_every(train_loader, print_freq, header):
        if args.model_type=="MLP" or args.model_type=="CNN" or args.model_type=="ViT":
            samples = data
            samples = samples.to(args.device, non_blocking=True)
        elif args.model_type=="Fusion":
            samples_img, samples_meta = data
            samples_img = samples_img.to(args.device, non_blocking=True)
            samples_meta = samples_meta.to(args.device, non_blocking=True)
            
        targets = targets.to(args.device, non_blocking=True)
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if args.model_type=="MLP" or args.model_type=="CNN" or args.model_type=="ViT":
                outputs = model(samples)
            elif args.model_type=="Fusion":
                outputs = model(samples_img, samples_meta)
            
            loss = criterion(outputs, targets)
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
            
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

if __name__ == '__main__':
    args = parse_args()
    output_dir = Path(args.output_dir).joinpath('{}/{}/{}/'.format(args.data_set, args.model_type, args.loss_fn))
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

