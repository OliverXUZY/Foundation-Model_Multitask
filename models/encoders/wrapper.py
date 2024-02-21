from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .encoders import register
from ..modules import *

# @register('wrapper')
# class Wrapper(Module):
#     def __init__(self, enc, out_dim = 100):
#         super(Wrapper, self).__init__()
#         self.enc = enc
#         self.wrap = nn.Sequential(
#             nn.Linear(self.enc.get_out_dim(), out_dim),
#             nn.ReLU(),
#         )
#         self.out_dim = out_dim
    
#     def get_out_dim(self):
#         return self.out_dim
    
#     def forward(self, x):
#         assert x.dim() == 4     # [B, C, H, W]
#         x = self.enc(x)
#         x = x.float()
#         wrap = self.wrap(x)
#         return wrap


# __all__ = ['wrapper', 'TwoLayersNN']

# @register('TwoLayersNN')
# class TwoLayersNN(Module):
#     def __init__(self, in_dim, hidden_dim = 2048, out_dim = 1024):
#         super(TwoLayersNN, self).__init__()
#         self.layers = nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, out_dim),
#                 nn.ReLU(),
#             )
#         self.out_dim = out_dim
    
#     def get_out_dim(self):
#         return self.out_dim
    
#     def forward(self, x):
#         assert x.dim() == 2    # [B, D]
#         x = x.float()
#         return self.layers(x)


# @register('wrapper')
# class Wrapper(Module):
#     def __init__(self, enc, wrap):
#         super(Wrapper, self).__init__()
#         self.enc = enc
#         self.wrap = wrap
    
#     def get_out_dim(self):
#         return self.wrap.get_out_dim()
    
#     def forward(self, x):
#         assert x.dim() == 4     # [B, C, H, W]
#         x = self.enc(x)
#         x = x.float()
#         wrap = self.wrap(x)
#         return wrap

__all__ = ['wrapper', 'OneLayerNN', 'TwoLayersNN', 'ResNetBlock']


@register('OneLayerNN')
class OneLayerNN(Module):
    def __init__(self, in_dim, out_dim = 100):
        super(OneLayerNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        assert x.dim() == 2    # [B, D]
        x = x.float()
        return self.layers(x)

@register('TwoLayersNN')
class TwoLayersNN(Module):
    def __init__(self, in_dim, hidden_dim = 2048, out_dim = 1024):
        super(TwoLayersNN, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU(),
            )
        self.out_dim = out_dim
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        assert x.dim() == 2    # [B, D]
        x = x.float()
        return self.layers(x)

class TwoLayersResNetBlock(Module):
    def __init__(self, in_dim, hidden_dim = 2048):
        super(TwoLayersResNetBlock, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),
                nn.ReLU(),
            )
        self.out_dim = in_dim

    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        assert x.dim() == 2
        x = x.float()

        return x + self.layers(x)


@register('twoLayersResNet')
def twoLayersResNet(in_dim):
    return TwoLayersResNetBlock(in_dim)


@register('wrapper')
class Wrapper(Module):
    def __init__(self, enc, wrap):
        super(Wrapper, self).__init__()
        self.enc = enc
        self.wrap = wrap
    
    def get_out_dim(self):
        return self.wrap.get_out_dim()
    
    def forward(self, x):
        assert x.dim() == 4     # [B, C, H, W]
        x = self.enc(x)
        x = x.float()
        wrap = self.wrap(x)
        return wrap







