import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *


class Cifar100(Dataset):
  def __init__(self, root, split='train', size=32, n_view=1, transform=None):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views. Default: 1
      transform (str): data augmentation. Default: None
    """
    super(Cifar100, self).__init__()
    
    split_dict = {'train': 'train',             # standard train
                  'trainval': 'trainval',       # standard train + val
                  'meta-train': 'train',        # meta-train
                  'meta-val': 'val',            # meta-val
                  'meta-trainval': 'trainval',  # meta-train + meta-val
                  'meta-test': 'test',          # meta-test
                 }
    split_tag = split_dict[split]

    split_file = os.path.join(root, split_tag + '.pickle')
    assert os.path.isfile(split_file)
    with open(split_file, 'rb') as f:
      pack = pickle.load(f, encoding='latin1')
    data, label = pack['data'], pack['labels']

    data = [Image.fromarray(x) for x in data]
    label = np.array(label)
    label_key = sorted(np.unique(label))
    label_map = dict(zip(label_key, range(len(label_key))))
    new_label = np.array([label_map[x] for x in label])
    
    self.root = root
    self.split_tag = split_tag
    self.size = size
    
    self.data = data
    self.label = new_label
    self.n_class = len(label_key)

    self.statistics = {'mean': [0.5071, 0.4867, 0.4408],
                       'std':  [0.2675, 0.2565, 0.2761]}
    transform = get_transform(transform, size, self.statistics)
    self.transform = MultiViewTransform(transform, n_view)
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])            # [V, C, H, W]
    label = self.label[index]
    return image, label


class MetaCifar100(Cifar100):
  def __init__(self, root, split='meta-train', size=32, 
               n_view=1, n_meta_view=1, share_query=False,
               transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views of image. Default: 1
      n_meta_view (int): number of augmented views of task. Default: 1
      share_query (bool): True if use distinct query set for each meta-view. 
        Default: False
      transform (str): training data augmentation. Default: None
      val_transform (str): validation data augmentation. Default: None
      n_batch (int): number of mini-batches per epoch. Default: 200
      n_episode (int): number of episodes (tasks) per mini-batch. Default: 4
      n_way (int): number of categories per episode. Default: 5
      n_shot (int): number of training (support) samples per category. 
        Default: 1
      n_query (int): number of validation (query) samples per category. 
        Default: 15
    """
    super(MetaCifar100, self).__init__(root, split, size, n_view, transform)
    
    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.n_shot_view = self.n_meta_view = n_meta_view
    if share_query:
      self.n_query_view = 1
    else:
      self.n_query_view = n_meta_view

    self.catlocs = tuple()
    for cat in range(self.n_class):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

    self.val_transform = get_transform(val_transform, size, self.statistics)

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    s, q = self.n_shot, self.n_query
    sv, qv = self.n_shot_view, self.n_query_view
    shot, query = tuple(), tuple()
    
    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    for c in cats:
      idx = np.random.choice(self.catlocs[c], sv * s + qv * q, replace=False)
      s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
      c_shot = torch.stack([self.transform(self.data[i]) for i in s_idx])
      c_query = torch.stack([self.val_transform(self.data[i]) for i in q_idx])
      c_shot = c_shot.view(sv, s, *c_shot.shape[-4:])
      c_query = c_query.view(qv, q, *c_query.shape[-3:])
      shot += (c_shot,)
      query += (c_query,)
    
    shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]
    query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]
    cats = torch.from_numpy(cats)
    return shot, query, cats


@register('cifar-fs')
class CifarFS(Cifar100):
  pass

@register('meta-cifar-fs')
class MetaCifarFS(MetaCifar100):
  pass


@register('fc100')
class FC100(Cifar100):
  pass


@register('meta-fc100')
class MetaFC100(MetaCifar100):
  pass
