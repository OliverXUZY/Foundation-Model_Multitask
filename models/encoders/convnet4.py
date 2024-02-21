from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .encoders import register
from ..modules import *


__all__ = ['convnet4', 'wide_convnet4']


class ConvBlock(Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()

    self.conv = Conv2d(in_channels, out_channels, 3, 1, padding=1)
    self.bn = BatchNorm2d(out_channels)
    self.relu = ReLU(inplace=True)
    self.pool = MaxPool2d(2)
    
  def forward_cp(self, x):
    self.is_first_pass(not torch.is_grad_enabled())
    return self.forward(x)

  def forward(self, x, hz=False):
    z1 = self.conv(x)
    h, z = (x,), (z1,)
    if isinstance(z1, (tuple, list)):
      z1 = torch.cat(z1)
    x = self.relu(self.bn(z1))
    x = self.pool(x)
    if hz:
      return x, h, z
    else:
      return x


class ConvNet4(Module):
  def __init__(self, hid_dim, out_dim):
    super(ConvNet4, self).__init__()

    self.conv1 = ConvBlock(3, hid_dim)
    self.conv2 = ConvBlock(hid_dim, hid_dim)
    self.conv3 = ConvBlock(hid_dim, hid_dim)
    self.conv4 = ConvBlock(hid_dim, out_dim)

  def get_out_dim(self, scale=25):
    return self.out_dim * scale

  def _forward_gen(self, module):
    def forward_cp(*state):
      return module.forward_cp(state[0])
    return forward_cp

  def forward(self, x, hz=False):
    split = isinstance(x, (tuple, list))
    if split:
      sizes = [len(k) for k in x]
      x = torch.cat(x)

    if self.efficient and self.training:
      dummy = torch.ones(1, requires_grad=True).to(x.device)
      x = checkpoint(self._forward_gen(self.conv1), x, dummy)
      x = checkpoint(self._forward_gen(self.conv2), x)
      x = checkpoint(self._forward_gen(self.conv3), x)
    else:
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)

    if split:
      x = torch.split(x, sizes)

    if hz:
      with torch.enable_grad():
        x, h, z = self.conv4(x, hz)
      x = x.flatten(1)
      return x, h, z
    else:
      x = self.conv4(x)
      x = x.flatten(1)
      return x


@register('convnet4')
def convnet4():
  return ConvNet4(32, 32)


@register('wide-convnet4')
def wide_convnet4():
  return ConvNet4(64, 64)