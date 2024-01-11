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

from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

import utils
import copy
from pathlib import Path
from tqdm import tqdm


def compute_scores(outputs, targets, threshold=0.45):
    
    prob = outputs >= threshold
    label = targets >= threshold
    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()
    
    return TP, TN, FP, FN


def compute_metrics(TP, TN, FP, FN, weight):
    
    # macro true skill score
    macro_recall = TP / (TP + FN + 1e-8)
    macro_specificity = TN / (TN + FP + 1e-8)
    macro_tss = torch.mean(macro_recall + macro_specificity - 1)
    
    # micro true skill score
    micro_recall = TP.sum() / (TP.sum() + FN.sum())
    micro_specificity = TN.sum() / (TN.sum() + FP.sum())
    micro_tss = micro_recall + micro_specificity - 1
    
    # weighted true skill score
    weighted_recall = weight @ macro_recall
    weighted_specificity = weight @ macro_specificity
    weighted_tss = weighted_recall + weighted_specificity - 1
    
    # true skill score per class
    
    return macro_tss.item(), micro_tss.item(), weighted_tss.item()


def compute_metrics_per_cls(TP, TN, FP, FN):
    
    # macro true skill score
    recall = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    
    #P = TP + FN
    #N = FP + TN
    #prevalence = P / (P + N + 1e-8)
    
    # true skill score per class
    tss = recall + specificity - 1
    
    return recall, specificity, tss
    

def evaluate(model, eval_loader, criterion, args):
    
    TP, TN, FP, FN = 0., 0., 0., 0.
    label_cnt = 0.
    eval_loss = []
    model.eval()
    with torch.no_grad():
        for data, targets in tqdm(eval_loader, file=sys.stdout):
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
            label_cnt += targets.sum(0).float()
            
            loss = criterion(outputs, targets)
            eval_loss.append(loss.item())
            
            if args.thres_method == 'adaptive':
                thresholds = torch.load(args.output_dir.joinpath('thresholds_train.pth'))
                thresholds = torch.tensor(thresholds).to(args.device)
            elif args.thres_method == 'global':
                thresholds = args.threshold
                
            tp, tn, fp, fn = compute_scores(torch.sigmoid(outputs), targets, thresholds)
            TP += tp
            TN += tn
            FP += fp
            FN += fn
        
        eval_loss = torch.tensor(eval_loss).mean().item()
        weight = label_cnt / label_cnt.sum()
        macro_tss, micro_tss, weighted_tss = compute_metrics(TP, TN, FP, FN, weight=weight)
        
        if args.eval:
            recall, spec, tss = compute_metrics_per_cls(TP, TN, FP, FN)
            species = []
            cdref2species = pd.read_csv('filelist/cdref2species.csv')
            for ref in list(map(float, args.classes)):
                temp = cdref2species.loc[cdref2species.cd_ref==ref]
                species.append(temp.nom_reconnu.values[0])
            df = pd.DataFrame({'species':species, 'train_cnt':args.train_label_cnt.values(), 'test_cnt': args.test_label_cnt.values(), 'recall': recall.detach().cpu(), 'specificity': spec.detach().cpu(), 'tss':tss.detach().cpu()})
            df.to_csv(args.output_dir.joinpath('{}_tss_per_cls.csv'.format(args.model_type)), index=False)
            print('Per class performance saved!')
        
        eval_stats = {'eval_loss': eval_loss, 'macro_tss': macro_tss, 'micro_tss': micro_tss, 'weighted_tss': weighted_tss}
        
        return eval_stats
            
            
            
            
            
            
    
