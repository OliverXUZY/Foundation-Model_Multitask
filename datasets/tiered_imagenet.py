import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *

import time
import logging
logger = logging.getLogger(__name__)

@register('tiered-imagenet')
class TieredImageNet(Dataset):
  def __init__(self, root, split='train', size=84, n_view=1, transform=None):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views. Default: 1
      transform (str): data augmentation. Default: None
    """
    super(TieredImageNet, self).__init__()
    
    split_dict = {'train': 'train',         # standard train
                  'val': 'train_phase_val', # standard val
                  'meta-train': 'train',    # meta-train
                  'meta-val': 'val',        # meta-val
                  'meta-test': 'test',      # meta-test
                 }
    split_tag = split_dict[split]

    split_file = os.path.join(root, split_tag + '_images.npz')
    print(split_file)
    label_file = os.path.join(root, split_tag + '_labels.pkl')
    assert os.path.isfile(split_file)
    assert os.path.isfile(label_file)
    data = np.load(split_file, allow_pickle=True)['images']
    data = data[:, :, :, ::-1]
    with open(label_file, 'rb') as f:
      label = pickle.load(f)['labels']

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

    self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    transform = get_transform(transform, size, self.statistics)
    self.transform = MultiViewTransform(transform, n_view)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])              # [V, C, H, W]
    label = self.label[index]
    return image, label


@register('meta-tiered-imagenet')
class MetaTieredImageNet(TieredImageNet):
  def __init__(self, root, split='meta-train', size=84, 
               n_view=1, n_meta_view=1, share_query=False,
               transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic = False,
               limited_class = None, select_class = None):
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
      deterministic: whether set images in dataset to be deterministic in each epoch
    """
    super(MetaTieredImageNet, self).__init__(root, split, size, n_view, transform)
    
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
    
    ##
    if limited_class:
      if isinstance(limited_class, float):
        if limited_class >= 1:
          num_classes = int(limited_class)
        else:
          num_classes = int(self.n_class*limited_class)
      else:
        raise ValueError("limited_class is none, but is neither an integer nor a float")
      self.n_class = np.random.choice(self.n_class, num_classes, replace=False)
    
    if select_class:
      indices = [  7,   8,  20,  22,  24,  26,  28,  30,  31,  35,  39,  41,  50,  60,
        108, 109, 113, 115, 118, 119, 121, 122, 123, 124, 126, 127, 128, 129,
        136, 137, 139, 140, 145, 146, 160, 161, 162, 163, 164, 165, 166, 167,
        168, 169, 171, 172, 173, 174, 176, 177, 178, 180, 181, 183, 184, 185,
        186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 201,
        202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 213, 214, 215, 216,
        217, 218, 219, 222, 223, 225, 227, 229, 230, 231, 232, 234, 235, 237,
        238, 239, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254,
        255, 256, 260, 262, 263, 264, 265, 266, 267, 269, 270, 271, 273, 276,
        277, 278, 283, 284, 288, 290, 291, 294, 295, 296, 297, 298, 299, 300,
        301, 302, 303, 304, 305, 306, 308, 309, 310, 312, 314, 315, 317, 318,
        320, 321, 323, 326, 327, 328, 330, 331, 332, 333, 335, 336, 337, 340,
        341, 342, 343, 344, 347, 348, 349]
        
      self.n_class = indices
    print("limited accessed classes to {}".format(self.n_class))

    self.val_transform = get_transform(val_transform, size, self.statistics)

    self.deterministic = deterministic

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    if self.deterministic:
      np.random.seed(index)  ## add for control # of tasks and # of images
    s, q = self.n_shot, self.n_query
    sv, qv = self.n_shot_view, self.n_query_view
    shot, query = tuple(), tuple()
    
    replace =  False
    if not isinstance(self.n_class, int) and len(self.n_class) < self.n_way:
      replace = True
    cats = np.random.choice(self.n_class, self.n_way, replace=replace)
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
  
@register('seq-meta-tiered-imagenet')
class SeqMetaTieredImageNet(MetaTieredImageNet):
  def __init__(self, root, split='meta-train', size=84, 
               n_view=1, n_meta_view=1, share_query=False,
               transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic = False):
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
      deterministic: whether set images in dataset to be deterministic in each epoch
    """
    super().__init__(root, split, size, 
               n_view, n_meta_view, share_query,
               transform, val_transform,
               n_batch, n_episode, n_way, n_shot, n_query, deterministic)
    
    # in seq loader, idx within each class is recorded
    self.idx_within_class = np.zeros(self.n_class)

  def __getitem__(self, index):
    if self.deterministic:
      np.random.seed(index)  ## add for control # of tasks and # of images
    s, q = self.n_shot, self.n_query
    sv, qv = self.n_shot_view, self.n_query_view
    shot, query = tuple(), tuple()
    print("zhuoyan: this is outer index======: ", index)

    if (index + 1) * self.n_way % self.n_class <= index * self.n_way % self.n_class:
        cats = (np.concatenate([np.arange(index* self.n_way % self.n_class, self.n_class), np.arange((index + 1) * self.n_way % self.n_class)]))
    else:
        cats = (np.arange(index * self.n_way % self.n_class, (index+ 1) * self.n_way % self.n_class))
      
    print("This is cats: ====", cats)
    
    # cats = np.random.choice(self.n_class, self.n_way, replace=False)
    for c in cats:
      batch_size = sv * s + qv * q
      total_size = len(self.catlocs[c])
      index_per_class = self.idx_within_class[c]
      print("zhuoyan: this is class {} with index_per_class======: {}".format(c, index_per_class))

      if (index_per_class + 1) * batch_size % total_size <= index_per_class * batch_size % total_size:
        idx = self.catlocs[c][np.concatenate([np.arange(index_per_class* batch_size % total_size, total_size, dtype=int), np.arange((index_per_class + 1) * batch_size % total_size, dtype=int)])]
      else:
        print("zhuoyannn: {} to {}".format(index_per_class * batch_size % total_size, (index_per_class+ 1) * batch_size % total_size) )
        print(np.arange(index_per_class * batch_size % total_size, (index_per_class+ 1) * batch_size % total_size, dtype=int))
        idx = self.catlocs[c][np.arange(index_per_class * batch_size % total_size, (index_per_class+ 1) * batch_size % total_size, dtype=int)]
      
      # idx = np.random.choice(self.catlocs[c], sv * s + qv * q, replace=False)
      print(idx)
      self.idx_within_class[c] += 1

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

######################################################################################################################
########################################## vision language dataset
from torchvision.datasets import ImageFolder

data_root = "/datadrive/datasets"
with open(os.path.join(data_root,'classnames.txt')) as f:
    lines = [line.rstrip() for line in f]

class_to_name = {}
for line in lines:
    s_id = line.find(' ')
    class_to_name[line[:s_id]] = line[s_id+1:]

@register('vl-meta-tiered-imagenet') 
class VLMetaTieredImageNet(Dataset):
  def __init__(self, root, split='train', size=224, transform=None,
               n_batch=200, n_episode=4, n_way=15, n_query=15, deterministic = False):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      transform (str): data augmentation. Default: None
    """
    super(VLMetaTieredImageNet, self).__init__()

    split_dict = {'train': 'train',        # standard train
                  'val': 'val',            # standard val
                  'test': 'test',          # standard test
                  'meta-train': 'train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
    split_tag = split_dict.get(split) or split
    split_dir = '{}/{}'.format(root, split_tag)
    print(split_dir)
    assert os.path.isdir(split_dir)
    
    self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    self.transform = get_transform(transform, size, self.statistics)

    self.dataset = ImageFolder(root = split_dir, transform = self.transform)

    idx_to_name = {}
    for c in self.dataset.class_to_idx:
        idx_to_name[self.dataset.class_to_idx[c]] = class_to_name[c]
    self.n_class = len(idx_to_name)
    self.label_idx_to_name = idx_to_name

    
    ### sampling part
    print("start sampling part dataset")
    ##### cache label file since it's time consuming
    cache_label_file = os.path.join(root,"cached_{}_labels_vl-tiered-imagenet.npy".format(split_tag))
    if os.path.exists(cache_label_file):
      start = time.time()
      self.label = np.load(cache_label_file)
      print(
          f"Loading labels from cached file {cache_label_file} [took %.3f s]", time.time() - start
      )
    else:
      print(f"Creating labels from dataset file at {root}")
      start = time.time()
      self.label = np.array([target for _, target in self.dataset])
      np.save(cache_label_file, self.label)
      # ^ This seems to take forever (but 5 mins at my laptop) so I want to investigate why and how we can improve.
      print(
          "Saving labels into cached file %s [took %.3f s]", cache_label_file, time.time() - start
      )

    self.catlocs = tuple()
    for cat in range(self.n_class):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_query = n_query
    self.deterministic = deterministic
  
  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    if self.deterministic:
      np.random.seed(index) 
    q = self.n_query
    query = tuple()

    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    label_names = []
    for c in cats:
      label_names.append(self.label_idx_to_name[c])
      idx = np.random.choice(self.catlocs[c], q, replace=False) 
      c_query = torch.stack([self.dataset[i][0] for i in idx])  # [q, C, H ,W] [3, 3, 224, 224]
      query += (c_query,)
    query = torch.cat(query)    # [Y * Q, C, H, W] 
    cats = torch.from_numpy(cats)
    
    return query, cats, label_names
