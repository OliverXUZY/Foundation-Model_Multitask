
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import clip

from .encoders import register
from ..modules import *

__all__ = ['RN50']

@register('RN50')
class RN50(Module):
    '''
    ResNet50 encoder pre-trained by CLIP
    '''
    def __init__(self):
        super(RN50, self).__init__()

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








