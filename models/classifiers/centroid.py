import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifiers import register


@register('centroid')
class CentroidClassifier(nn.Module):
  def __init__(self, in_dim, n_way, temp=1., learn_temp=False):
    super(CentroidClassifier, self).__init__()

    self.centroid = nn.Parameter(torch.empty(n_way, in_dim))
    nn.init.kaiming_uniform_(self.centroid, a=math.sqrt(5))

    self.temp = temp
    if learn_temp:
      self.temp = nn.Parameter(torch.tensor(temp))

  def forward(self, x):
    #############################################
    # B: mini-batch size
    # D: input / feature dimension
    # Y: number of ways / categories
    #############################################
    assert x.dim() == 2                             # [B, D]
    x = F.normalize(x, dim=-1)                      # [B, D]
    centroid = F.normalize(self.centroid, dim=-1)   # [Y, D]
    x = torch.mm(x, centroid.T)                     # [B, Y]
    x = x * self.temp
    return x


@register('fs-centroid')
class FSCentroidClassifier(nn.Module):
  def __init__(self, in_dim, n_way=None, temp=1., learn_temp=False):
    super(FSCentroidClassifier, self).__init__()

    self.temp = temp
    if learn_temp:
      self.temp = nn.Parameter(torch.tensor(temp))

  def forward(self, s, q):
    #############################################
    # E: number of episodes / tasks
    # SV: number of shot views per task
    # QV: number of query views per task
    # V: number of views per image
    # Y: number of ways / categories per task
    # S: number of shots per category
    # Q: number of queries per category
    # D: input / feature dimension
    #############################################
    assert s.dim() == 6                             # [SV, E, Y, S, V, D]
    assert q.dim() == 4                             # [QV, E, Y * Q, D]
    assert q.size(0) in [1, s.size(0)]
    
    centroid = torch.mean(s, dim=(-3, -2))          # [SV, E, Y, D]
    centroid = F.normalize(centroid, dim=-1)        # [SV, E, Y, D]
    centroid = centroid.transpose(-2, -1)           # [SV, E, D, Y]
    q = F.normalize(q, dim=-1)                      # [QV, E, Y * Q, D]
    logits = torch.matmul(q, centroid)              # [SV, E, Y * Q, Y]
    logits = logits * self.temp
    return logits