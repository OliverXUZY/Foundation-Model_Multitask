import torch.nn as nn
import torch.nn.functional as F

from .projectors import register


@register('linear')
class LinearProjection(nn.Module):
  def __init__(self, in_dim, dim=128):
    super(LinearProjection, self).__init__()

    self.dim = dim
    self.fc = nn.Linear(in_dim, dim)

  def get_out_dim(self):
    return self.dim

  def forward(self, x):
    x = F.normalize(self.fc(x), dim=-1)
    return x