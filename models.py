from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import torch

import timm
from timm.models import create_model
from timm.scheduler import CosineLRScheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict


class Mlp_CBNA(nn.Module):
    def __init__(self, args, out_dim, n_meta_dim=[512, 128], act_layer=nn.ReLU):
        super(Mlp_CBNA, self).__init__()
        
        self.meta = nn.Sequential(
            nn.Linear(args.n_meta_features, n_meta_dim[0]),
            nn.BatchNorm1d(n_meta_dim[0]),
            act_layer(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(n_meta_dim[0], n_meta_dim[1]),
            nn.BatchNorm1d(n_meta_dim[1]),
            act_layer(inplace=True),
        )
        in_ch = n_meta_dim[-1]
        self.fc = nn.Linear(in_ch, out_dim, bias=True)

    def forward(self, x):
        
        out = self.meta(x)
        out = self.fc(out)
            
        return out
    

class Vit_CBNA(nn.Module):
    def __init__(self, args, out_dim, n_meta_dim=[512, 128], act_layer=nn.GELU, pretrained=False):
        super(Vit_CBNA, self).__init__()
        
        self.base_model = create_model(
        'deit3_base_patch16_224',
        pretrained=pretrained,
        drop_path_rate = args.drop_path,
        num_classes=args.num_classes
        )
        

    def forward(self, x):
        
        out_img = self.base_model(x)
        out = out_img
            
        return out
    
    
class Resnet_CBNA(nn.Module):
    def __init__(self, args, out_dim, n_meta_dim=[512, 128], act_layer=nn.GELU, pretrained=False):
        super(Resnet_CBNA, self).__init__()
        
        self.base_model = create_model(
        'resnet50',
        pretrained=pretrained,
        drop_path_rate = args.drop_path,
        num_classes=args.num_classes
        )
        

    def forward(self, x):
        
        out_img = self.base_model(x)
        out = out_img
            
        return out
    

class Fusion_CBNA(nn.Module):
    def __init__(self, args, out_dim, n_meta_dim=[512, 128], act_layer=nn.GELU, pretrained=False):
        super(Fusion_CBNA, self).__init__()
        
        self.base_model = create_model(
        'resnet50',
        pretrained=pretrained,
        drop_path_rate = args.drop_path,
        num_classes=args.num_classes
        )
        
        in_ch = self.base_model.fc.in_features
        self.meta = nn.Sequential(
            nn.Linear(args.n_meta_features, n_meta_dim[0]),
            nn.BatchNorm1d(n_meta_dim[0]),
            act_layer(),
            nn.Dropout(p=0.3),
            nn.Linear(n_meta_dim[0], n_meta_dim[1]),
            nn.BatchNorm1d(n_meta_dim[1]),
            act_layer(),
        )
        in_ch += n_meta_dim[-1]

        self.base_model.fc = nn.Identity()
        self.myfc = nn.Linear(in_ch, out_dim, bias=True)

    def forward(self, x_img, x_meta):
        
        out_img = self.base_model(x_img)
        out_meta = self.meta(x_meta)
        out = torch.cat((out_img, out_meta), dim=1)
        out = self.myfc(out)
            
        return out