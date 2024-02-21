import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
  def __init__(self):
    super(Module, self).__init__()
    self.efficient = False
    self.first_pass = True

  def go_efficient(self, mode=True):
    """ Switches on / off gradient checkpointing. """
    self.efficient = mode
    for m in self.children():
      if isinstance(m, Module):
        m.go_efficient(mode)

  def is_first_pass(self, mode=True):
    """ Determines whether to update batchnorm statistics. """
    self.first_pass = mode
    for m in self.children():
      if isinstance(m, Module):
        m.is_first_pass(mode)


class Conv2d(nn.Conv2d, Module):
  def __init__(self, in_channels, out_channels, kernel_size, 
               stride=1, padding=0, bias=True):
    super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, 
                                 stride, padding, bias=bias)

  def forward(self, x):
    split = isinstance(x, (tuple, list))
    if split:
      sizes = [len(k) for k in x]
      x = torch.cat(x)
    x = super(Conv2d, self).forward(x)
    if split:
      x = torch.split(x, sizes)
    return x


class Linear(nn.Linear, Module):
  def __init__(self, in_features, out_features, bias=True):
    super(Linear, self).__init__(in_features, out_features, bias=bias)

  def forward(self, x):
    split = isinstance(x, (tuple, list))
    if split:
      sizes = [len(k) for k in x]
      x = torch.cat(x)
    x = super(Linear, self).forward(x)
    if split:
      x = torch.split(x, sizes)
    return x


class BatchNorm2d(nn.BatchNorm2d, Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
               track_running_stats=True):
    super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, 
                                      track_running_stats)

  def forward(self, x):
    split = isinstance(x, (tuple, list))
    if split:
      sizes = [len(k) for k in x]
      x = torch.cat(x)

    self._check_input_dim(x)
    exp_avg_factor = 0.
    if self.first_pass and self.training and self.track_running_stats:
      self.num_batches_tracked += 1
      if self.momentum is None:
        exp_avg_factor = 1. / float(self.num_batches_tracked)
      else:
        exp_avg_factor = self.momentum

    x = F.batch_norm(x, self.running_mean, self.running_var, 
                     self.weight, self.bias,
                     self.training or not self.track_running_stats, 
                     exp_avg_factor, self.eps)

    if split:
      x = torch.split(x, sizes)
    return x


class ReLU(nn.ReLU, Module):
  def __init__(self, inplace=False):
    super(ReLU, self).__init__(inplace)

  def forward(self, x):
    split = isinstance(x, (tuple, list))
    if split:
      sizes = [len(k) for k in x]
      x = torch.cat(x)
    x = super(ReLU, self).forward(x)
    if split:
      x = torch.split(x, sizes)
    return x


class LeakyReLU(nn.LeakyReLU, Module):
  def __init__(self, negative_slope=0.01, inplace=False):
    super(LeakyReLU, self).__init__(negative_slope, inplace)

  def forward(self, x):
    split = isinstance(x, (tuple, list))
    if split:
      sizes = [len(k) for k in x]
      x = torch.cat(x)
    x = super(LeakyReLU, self).forward(x)
    if split:
      x = torch.split(x, sizes)
    return x


class Sequential(nn.Sequential, Module):
  def __init__(self, *args):
    super(Sequential, self).__init__(*args)

  def forward(self, x):
    return super(Sequential, self).forward(x)