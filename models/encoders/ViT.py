
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import clip
import os

from .encoders import register
from ..modules import *
from . import mocov3_vits as vits

__all__ = ['clip_ViT-B32', 
           'dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14', 
           'dino_vitb16',
           'torchvision_vit_b_32', 'mocov3_vit']

@register('clip_ViT-B32')
class clip_ViTB32(Module):
    '''
    ViT encoder pre-trained by CLIP
    '''
    def __init__(self):
        super(clip_ViTB32, self).__init__()

        self.out_dim = 512
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, 512]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model.encode_image(x)


class dinov2_vit(Module):
    name_to_embd_dim = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        'dinov2_vitl14': 1024, 
        'dinov2_vitg14': 1536
    }
    '''
    ViT encoder pre-trained by DINOv2
    '''
    def __init__(self, model_name):
        super(dinov2_vit, self).__init__()

        self.out_dim = dinov2_vit.name_to_embd_dim[model_name]
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, embd_dim]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)


@register('dinov2_vits14')
def dinov2_vits14():
  return dinov2_vit("dinov2_vits14")

@register('dinov2_vitb14')
def dinov2_vitb14():
  return dinov2_vit("dinov2_vitb14")

@register('dinov2_vitl14')
def dinov2_vitl14():
  return dinov2_vit("dinov2_vitl14")

@register('dinov2_vitg14')
def dinov2_vitg14():
  return dinov2_vit("dinov2_vitg14")


class dino_vit(Module):
    name_to_embd_dim = {
        "dino_vitb16": 768,
    }
    '''
    ViT encoder pre-trained by DINO
    '''
    def __init__(self, model_name):
        super(dino_vit, self).__init__()

        self.out_dim = dino_vit.name_to_embd_dim[model_name]
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, embd_dim]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)

@register('dino_vitb16')
def dino_vitb16():
  return dino_vit("dino_vitb16")


@register('torchvision_vit_b_32')
class torchvision_vit_b_32(Module):
    '''
    RN50 encoder pre-trained by torchvision
    '''
    def __init__(self):
        super(torchvision_vit_b_32, self).__init__()

        self.out_dim = 768

        weights = torch.hub.load("pytorch/vision", "get_weight", name="ViT_B_32_Weights.IMAGENET1K_V1")

        self.model = torch.hub.load("pytorch/vision", "vit_b_32", weights=weights)
        self.model.heads = nn.Identity() # replace last nn.Linear(768, 1000) to Identity()

    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, embd_dim]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)

@register('mocov3_vit')# input image size for moco is still [3,224,224]: https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/main_lincls.py#L372
class mocov3_vit(Module):
    name_to_embd_dim = {
        "vit_small": 384,
        "vit_base": 768,
    }
    '''
    ViT encoder pre-trained by moco v3
    '''
    def __init__(self, arch, ckpt_path):
        super(mocov3_vit, self).__init__()
        self.model = vits.__dict__[arch]()
        self.model.head = nn.Identity() # replace last nn.Linear(D (368), 1000) to Identity()

        self.out_dim = mocov3_vit.name_to_embd_dim[arch]

        ## load state_dict of moco v3
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'head'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = self.model.load_state_dict(state_dict, strict=False)
        # print("zhuoyan===", msg)
        # print(set(msg.missing_keys))
        # print({"%s.weight" % 'head', "%s.bias" % 'head'})
        # assert set(msg.missing_keys) == {"%s.weight" % 'head', "%s.bias" % 'head'}

        print("=> loaded pre-trained model '{}'".format(ckpt_path))
    
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, D]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)
