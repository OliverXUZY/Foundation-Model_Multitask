import os

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *


@register('cub200')
class CUB200(Dataset):
  def __init__(self, root, split='train', size=84, n_view=1, transform=None):
    super(CUB200, self).__init__()
    
    split_dict = {'train': 'train',      # standard train
                  'meta-train': 'train', # meta-train
                  'meta-val': 'val',     # meta-val
                  'meta-test': 'test',   # meta-test
                 }
    split_tag = split_dict[split]

    split_file = os.path.join(root, 'fs-splits', split_tag + '.csv')
    assert os.path.isfile(split_file)
    with open(split_file, 'r') as f:
      pairs = [x.strip().split(',') 
                for x in f.readlines() if x.strip() != '']

    data, label = [x[0] for x in pairs], [int(x[1]) for x in pairs]
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

    self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std':  [0.229, 0.224, 0.225]}   # ImageNet statistics
    self.transform = get_transform(transform, size, self.statistics)
    self.transform = MultiViewTransform(transform, n_view)

  def _load_image(self, index):
    image_path = os.path.join(self.root, 'images', self.data[index])
    assert os.path.isfile(image_path)
    image = Image.open(image_path).convert('RGB')
    return image

  def __len__(self):
    return len(self.label)

  def __getitem__(self, index):
    image = self.transform(self._load_image(index))       # [V, C, H, W]
    label = self.label[index]
    return image, label


@register('meta-cub200')
class MetaCUB200(CUB200):
  def __init__(self, root, split='meta-train', size=84, 
               n_view=1, n_meta_view=1, share_query=False,
               transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15):
    super(MetaCUB200, self).__init__(root, split, size, n_view, transform)
    
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
      c_shot = torch.stack(
        [self.transform(self._load_image(idx)) for i in s_idx])
      c_query = torch.stack(
        [self.val_transform(self._load_image(idx)) for i in q_idx])
      c_shot = c_shot.view(sv, s, *c_shot.shape[-4:])
      c_query = c_query.view(qv, q, *c_query.shape[-3:])
      shot += (c_shot,)
      query += (c_query,)
    
    shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]
    query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]
    cats = torch.from_numpy(cats)
    return shot, query, cats