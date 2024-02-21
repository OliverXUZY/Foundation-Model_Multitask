import os
import pickle

import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *


@register('omniglot')
class Omniglot(Dataset):
  def __init__(self, root, split='train', size=105, n_view=1, transform=None, numClass = None):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 105
      n_view (int): number of augmented views. Default: 1
      transform (str): data augmentation. Default: None
    """
    super(Omniglot, self).__init__()
    
    split_dict = {
                  'meta-train': 'train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
    split_tag = split_dict.get(split) or split
    # split_tag = split_dict[split]
    split_file = root

    background = True if split_tag=='train' else False
    dataset = torchvision.datasets.Omniglot(
        root=split_file, download=True,background=background, transform=transforms.ToTensor(),
    )

    ## duplicate single channel to 3 channels
    image, _ = dataset[0]
    original_size = image.shape[1]
    data = [torchvision.transforms.ToPILImage()(image.expand(3,original_size,original_size)) for image,_ in dataset]
    label = [lb for _,lb in dataset]

    if split_tag == 'train' and numClass:
      if numClass == 482:
        data = data[:9640]
        label = label[:9640]
      elif numClass == 241:
        data = data[:4820]
        label = label[:4820]
      elif numClass == 50:
        data = data[:1000]
        label = label[:1000]
      elif numClass == 10:
        data = data[:200]
        label = label[:200]
    
    if split_tag == 'val':
        data = data[:6020]
        label = label[:6020]
    elif split_tag == 'test':
        data = data[6020:]
        label = label[6020:]


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
    
    
    transform = get_transform(transform, size)
    self.transform = MultiViewTransform(transform, n_view)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])            # [V, C, H, W]
    label = self.label[index]
    return image, label


@register('meta-omniglot')
class MetaOmniglot(Omniglot):
  def __init__(self, root, split='meta-train', size=105, 
               n_view=1, n_meta_view=1, share_query=False,
               transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, numClass = None):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 105
      n_view (int): number of augmented views of image. Default: 1. Only for shot, used in MultiViewTransform.
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
    super(MetaOmniglot, self).__init__(root, split, size, n_view, transform, numClass)
    
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

    self.val_transform = get_transform(val_transform, size)

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    s, q = self.n_shot, self.n_query
    sv, qv = self.n_shot_view, self.n_query_view
    shot, query = tuple(), tuple()
    
    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    for c in cats:
      idx = np.random.choice(self.catlocs[c], sv * s + qv * q, replace=False)      # random choose n_shot*shot_view + n_query*query_view (1*1+15*1) images in each classes
      s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
      c_shot = torch.stack([self.transform(self.data[i]) for i in s_idx])          # [1,1,3,84,84] [S*SV, V, C, H ,W]
      c_query = torch.stack([self.val_transform(self.data[i]) for i in q_idx])     # [15,3,84,84] [Q*QV, C, H ,W]
      c_shot = c_shot.view(sv, s, *c_shot.shape[-4:])                              # [1,1,1,3,84,84] [SV, S, V, C, H ,W]
      c_query = c_query.view(qv, q, *c_query.shape[-3:])                           # [1,10,3,84,84] [QV, Q, C, H ,W]
      shot += (c_shot,)
      query += (c_query,)
    
    shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]                   # [1, 5, 1, 3, 84, 84] 
    query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]                      # [1, 50, 3, 84, 84]
    cats = torch.from_numpy(cats)
    return shot, query, cats


