from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .encoders import register
from ..modules import *


__all__ = ['resnet12', 'wide_resnet12']


def conv3x3(in_channels, out_channels):
  return Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False)


def conv1x1(in_channels, out_channels):
  return Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=False)


class Block(Module):
  def __init__(self, in_planes, planes):
    super(Block, self).__init__()

    self.conv1 = conv3x3(in_planes, planes)
    self.bn1 = BatchNorm2d(planes)

    self.conv2 = conv3x3(planes, planes)
    self.bn2 = BatchNorm2d(planes)

    self.conv3 = conv3x3(planes, planes)
    self.bn3 = BatchNorm2d(planes)

    self.res_conv = conv1x1(in_planes, planes)
    self.res_bn = BatchNorm2d(planes)

    self.relu = LeakyReLU(0.1, inplace=True)
    self.pool = nn.MaxPool2d(2)

  def forward_cp(self, x):
    self.is_first_pass(not torch.is_grad_enabled())
    return self.forward(x)

  def forward(self, x, hz=False):
    z1 = self.conv1(x)
    h1 = self.relu(self.bn1(z1))
    z2 = self.conv2(h1)
    h2 = self.relu(self.bn2(z2))
    z3 = self.conv3(h2)
    h, z = (x, h1, h2), (z1, z2, z3)
    if isinstance(x, (tuple, list)):
      x = torch.cat(x)
    if isinstance(z3, (tuple, list)):
      z3 = torch.cat(z3)
    x = self.relu(self.bn3(z3) + self.res_bn(self.res_conv(x)))
    x = self.pool(x)
    if hz:
      return x, h, z
    else:
      return x


class ResNet12(Module):
  def __init__(self, channels):
    super(ResNet12, self).__init__()

    self.layer1 = Block(3, channels[0])
    self.layer2 = Block(channels[0], channels[1])
    self.layer3 = Block(channels[1], channels[2])
    self.layer4 = Block(channels[2], channels[3])
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.out_dim = channels[-1]

    for m in self.modules():
      if isinstance(m, Conv2d):
        nn.init.kaiming_normal_(
          m.weight, mode='fan_out', nonlinearity='leaky_relu')

  def get_out_dim(self):
    return self.out_dim

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
      x = checkpoint(self._forward_gen(self.layer1), x, dummy)
      x = checkpoint(self._forward_gen(self.layer2), x)
      x = checkpoint(self._forward_gen(self.layer3), x)
    else:
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)

    if split:
      x = torch.split(x, sizes)
    
    if hz:
      with torch.enable_grad():
        x, h, z = self.layer4(x, hz)
      x = self.pool(x).flatten(1)
      return x, h, z
    else:
      x = self.layer4(x)
      x = self.pool(x).flatten(1)
      return x


@register('resnet12')
def resnet12():
  return ResNet12([64, 128, 256, 512])


@register('wide-resnet12')
def wide_resnet12():
  return ResNet12([64, 160, 320, 640])