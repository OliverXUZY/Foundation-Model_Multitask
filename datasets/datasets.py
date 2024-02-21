import os

import torch
from torch.utils.data import Sampler
import numpy as np


DEFAULT_ROOT = '/datadrive/datasets'
datasets = {}

def register(name):
  def decorator(cls):
    datasets[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if kwargs.get('root') is None:
    kwargs['root'] = os.path.join(DEFAULT_ROOT, name.replace('meta-', '').replace('seq-','').replace('vl-',''))  # update for unsup-meta-mini-imagenet
  dataset = datasets[name](**kwargs)
  return dataset


class BalancedRandomSampler(Sampler):
  """ Samples equal number of images from each category. """
  def __init__(self, label):
    label = np.array(label)
    self.n_items = len(label)
    catlocs = tuple()
    for c in range(max(label) + 1):
      catlocs += (np.argwhere(label == c).reshape(-1),)
    self.catlocs = np.array(catlocs).T

  def __len__(self):
    return self.n_items

  def __iter__(self):
    catlocs = np.random.permutation(self.catlocs)
    idx_list = catlocs.reshape(-1).tolist()
    return iter(idx_list)