import torch.nn as nn

from .projectors import register


@register('identity')
class IdentityProjection(nn.Module):
  def __init__(self, in_dim, dim=512):
    super(IdentityProjection, self).__init__()
    assert in_dim == dim
    self.dim = dim

  def get_out_dim(self):
    return self.dim

  def forward(self, x):
    return x