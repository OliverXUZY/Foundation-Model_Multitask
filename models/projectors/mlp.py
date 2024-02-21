import torch.nn as nn
import torch.nn.functional as F

from .projectors import register


@register('mlp')
class MLPProjection(nn.Module):
  def __init__(self, in_dim, dim=128):
    super(MLPProjection, self).__init__()

    self.dim = dim
    self.fc1 = nn.Linear(in_dim, in_dim)
    self.fc2 = nn.Linear(in_dim, dim)

  def get_out_dim(self):
    return self.dim

  def forward(self, x):
    x = F.relu(self.fc1(x), inplace=True)
    x = F.normalize(self.fc2(x), dim=-1)
    return x