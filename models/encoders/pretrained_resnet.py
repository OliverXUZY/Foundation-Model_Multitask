
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import clip

from .encoders import register
from ..modules import *

import torchvision

__all__ = ['clip_RN50', 'ResNet50_mocov2', 'torchvision_RN50', 'mocov3_RN50', 'torchvision_RN18']

@register('clip_RN50')
class clip_RN50(Module):
    '''
    ResNet50 encoder pre-trained by CLIP
    '''
    def __init__(self):
        super(clip_RN50, self).__init__()

        self.out_dim = 1024
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50", device=device)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, 1024]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model.encode_image(x)

@register('ResNet50_mocov2') # input image size for moco is still [3,224,224]: https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/main_lincls.py#L372
class ResNet50_mocov2(Module):
    '''
    ResNet50 encoder pre-trained by moco v2
    '''
    def __init__(self, ckpt_path):
        super(ResNet50_mocov2, self).__init__()
        self.model = torchvision.models.__dict__['resnet50']()
        self.model.fc = nn.Identity() # replace last nn.Linear(2048, 1000) to Identity()

        self.out_dim = 2048

        ## load state_dict of moco v2
        print("=> loading checkpoint '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # rename moco pre-trained keys
        state_dict = ckpt["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        self.model.load_state_dict(state_dict, strict=False)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, 2048]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)

@register('torchvision_RN50')
class torchvision_RN50(Module):
    '''
    RN50 encoder pre-trained by torchvision
    '''
    def __init__(self):
        super(torchvision_RN50, self).__init__()

        self.out_dim = 2048

        weights = torch.hub.load("pytorch/vision", "get_weight", name="ResNet50_Weights.IMAGENET1K_V2")

        self.model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)
        self.model.fc = nn.Identity() # replace last nn.Linear(2048, 1000) to Identity()

    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, embd_dim]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)


@register('mocov3_RN50')# input image size for moco is still [3,224,224]: https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/main_lincls.py#L372
class mocov3_RN50(Module):
    '''
    ViT encoder pre-trained by moco v3
    '''
    def __init__(self, ckpt_path):
        super(mocov3_RN50, self).__init__()
        self.model = torchvision.models.__dict__['resnet50']()
        self.model.fc = nn.Identity() # replace last nn.Linear(2048, 1000) to Identity()

        self.out_dim = 2048

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


@register('torchvision_RN18')
class torchvision_RN18(Module):
    '''
    RN18 encoder pre-trained by torchvision
    '''
    def __init__(self):
        super(torchvision_RN18, self).__init__()

        self.out_dim = 512

        weights = torch.hub.load("pytorch/vision", "get_weight", name="ResNet50_Weights.IMAGENET1K_V2")

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Identity() # replace last nn.Linear(2048, 1000) to Identity()

    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, embd_dim]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)
