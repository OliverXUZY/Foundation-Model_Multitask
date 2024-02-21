from collections import OrderedDict

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .encoders import register
from ..modules import *


__all__ = ['resnet18', 'wide_resnet18']


def conv3x3(in_channels, out_channels, stride=1):
  return Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
  return Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False)


class Block(Module):
  def __init__(self, in_planes, planes, stride):
    super(Block, self).__init__()
    
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = BatchNorm2d(planes)

    self.conv2 = conv3x3(planes, planes)
    self.bn2 = BatchNorm2d(planes)

    self.stride = stride
    if stride > 1:
      self.res_conv = conv1x1(in_planes, planes)
      self.res_bn = BatchNorm2d(planes)

    self.relu = ReLU(inplace=True)

  def forward_cp(self, x):
    self.is_first_pass(not torch.is_grad_enabled())
    return self.forward(x)

  def forward(self, x, hz=False):
    z1 = self.conv1(x)
    h1 = self.relu(self.bn1(z1))
    z2 = self.conv2(h1)
    h, z = (x, h1), (z1, z2)
    if isinstance(x, (tuple, list)):
      x = torch.cat(x)
    if isinstance(z2, (tuple, list)):
      z2 = torch.cat(z2)
    if self.stride > 1:
      x = self.res_bn(self.res_conv(x))
    x = self.relu(self.bn2(z2) + x)
    if hz:
      return x, h, z
    else:
      return x


class ResNet18(Module):
  def __init__(self, channels):
    super(ResNet18, self).__init__()

    self.conv0 = conv3x3(3, 64)
    self.bn0 = BatchNorm2d(64)
    self.relu = ReLU(inplace=True)

    self.layer1 = Block(64, channels[0], 1)
    self.layer2 = Block(channels[0], channels[1], 2)
    self.layer3 = Block(channels[1], channels[2], 2)
    self.layer4 = Block(channels[2], channels[3], 2)
    self.pool = AdaptiveAvgPool2d(1)
    self.out_dim = channels[3]

    for m in self.modules():
      if isinstance(m, Conv2d):
        nn.init.kaiming_normal_(
          m.weight, mode='fan_out', nonlinearity='relu')

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
      
    x = self.relu(self.bn0(self.conv0(x)))
    if self.efficient and self.training:
      x = checkpoint(self._forward_gen(self.layer1), x)
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


@register('resnet18')
def resnet18():
  return ResNet18([64, 128, 256, 512])


@register('wide-resnet18')
def wide_resnet18():
  return ResNet18([64, 160, 320, 640])