import os
import pickle
import time
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image

from .datasets import register
from .transforms import *
from torchvision.datasets import ImageFolder


import torch.nn.functional as F
from collections import defaultdict

def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600) # time in hours
  if t >= 60:
    return '{:.1f}m'.format(t / 60)   # time in minutes
  return '{:.1f}s'.format(t)

from sklearn.neighbors import KernelDensity
import numpy as np

def load_test_embeddings():
  data_root = "/datadrive/datasets/mini-imagenet"
  dataset_name = "mini-imagenet"
  split_tag = "test"
  encoder_name = "dinov2_vitb14"
  cached_file_test = os.path.join(
      data_root,
      "{}_{}_{}_image_representation.json".format(dataset_name, split_tag, encoder_name)
  )

  if os.path.exists(cached_file_test):
      start = time.time()
      with open(cached_file_test) as f:
          mini_Rep_test = json.load(f)
      
      print(
          "Loading image features from cached file {} [took {:.3f} s]".format(cached_file_test, time.time() - start)
      )
  else:
      print("didn't load image feature | test")

  test_emb = torch.tensor(list(map(
      lambda name: mini_Rep_test[name],  
      mini_Rep_test.keys()
  ))).type(torch.float16)
  test_emb = test_emb.view(-1, 768)
  return test_emb


# torch.manual_seed(239)
def select(size = 2000, embedding = None):
    torch.manual_seed(239)
    # Create a random permutation of indices
    permuted_indices = torch.randperm(embedding.size(0))

    # Select the first 2000 indices
    selected_indices = permuted_indices[:size]

    # Index into the original tensor to get the subsample
    subsampled_data = embedding[selected_indices]

    return subsampled_data

def get_kde_target():
  test_emb = load_test_embeddings()

  subsampled_test = select(1000, test_emb)

  bandwidth = 0.5
  embeddings_target = subsampled_test
  kde_target = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
  kde_target.fit(embeddings_target)

  print("kde_target_fit done")

  return embeddings_target, kde_target

def load_train_embeddings():
  domains = ["sketch_split", "clipart_split", "infograph_split", "painting_split", "quickdraw_split", "real_split"]
  cached_file = "domainNet_train_dinov2_vitb14_image_representation"

  domain_features_raw = {}
  domain_features = {}
  for domain in domains:
    data_root = "/datadrive/datasets/domainNet/{}/".format(domain)
    cached_file = os.path.join(
        data_root,
        "domainNet_train_dinov2_vitb14_image_representation.json"
    )

    if os.path.exists(cached_file):
        start = time.time()
        with open(cached_file) as f:
            rep_train = json.load(f)
        print(
            "Loading image features from cached file {} [took {:.3f} s]".format(domain, time.time() - start)
        )
    else:
        print("didn't load image feature | {}".format(domain))
        continue
    
    domain_features_raw[domain] = rep_train
    class_features = []
    for key in rep_train:
        class_fea = torch.tensor(rep_train[key]).view(-1, 768)
        class_features.append(class_fea)
    
    class_features = torch.concat(class_features)
    print(class_features.shape)
    
    domain_features[domain] = class_features

  return domain_features_raw, domain_features


def get_domain_features_domain_kdes():
  _, domain_features = load_train_embeddings()
  domain_subsampled = {}
  for key, val in domain_features.items():
    domain_subsampled[key] = select(2000, val)
    print("{} done, shape{}".format(key, domain_subsampled[key].shape))

  domain_kdes = {}
  for domain, features in domain_subsampled.items():
    bandwidth = 0.5
    domain_kdes[domain] = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    domain_kdes[domain].fit(features)

  print("kde_train_fit done")

  return domain_features, domain_kdes

def compute_density_ratios(embeddings_train, kde_target, kde_train):
  assert embeddings_train.shape[1] == 768, "embedding size must match"

  # Compute the log densities (since scikit-learn returns log densities)
  # print("score samples")
  # start = time.time()
  log_density_train_train = kde_train.score_samples(embeddings_train)
  # print("score train embeddings on train, took {}".format(time_str(time.time() - start)))

  # start = time.time()
  log_density_target_train = kde_target.score_samples(embeddings_train)
  # print("score train embeddings on test, took {}".format(time_str(time.time() - start)))

  # Convert log densities back to normal densities
  density_train_train = np.exp(log_density_train_train)
  density_target_train = np.exp(log_density_target_train)

  return density_target_train / density_train_train




@register('select-domain-net')
class selectDomainNet(Dataset):
  def __init__(self, root, split='train', size=224, transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic = False, 
               domains=['real_split', 'painting_split', 'sketch_split', 'infograph_split', 'clipart_split', 'quickdraw_split']):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      transform (str): data augmentation. Default: None
    """
    super().__init__()

    split_dict = {'train': 'train',        # standard train
                  'val': 'val',            # standard val
                  'test': 'test',          # standard test
                  'meta-train': 'train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
    # specify root on my own
    root = "/datadrive/datasets/domainNet"
    print("selected domains: ", domains)

    self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    
    self.transform = get_transform(transform, size, self.statistics)

    split_tag = split_dict.get(split) or split

    if split_tag == "train":
        n_class = 180
    elif split_tag == "val":
      n_class = 65
    else:
      n_class = 100
    self.n_class = n_class
  
    self.datasets = {}  # each pair is domain: dataset
    self.catlocs_all_domain = {} # each pair is domain: self.catlocs
    self.domains = domains
    for domain in domains:
      root = "/datadrive/datasets/domainNet/{}".format(domain)
      
      split_dir = '{}/{}'.format(root, split_tag)
      print("for domain {}, split_dir is {}".format(domain,split_dir))
      assert os.path.isdir(split_dir)

      self.datasets[domain] = ImageFolder(root = split_dir)

      ### sampling part
      print("start sampling part dataset")
      ##### cache label file since it's time consuming
      cache_label_file = os.path.join(root,"cached_{}_labels_domainNet.npy".format(split_tag))
      if os.path.exists(cache_label_file):
        start = time.time()
        label = np.load(cache_label_file)
        print(
            "Loading labels from cached file {} [took {:.3f} s]".format(cache_label_file, time.time() - start)
        )
      else:
        print(f"Creating labels from dataset file at {root}")
        start = time.time()
        label = np.array([target for _, target in self.datasets[domain]])
        np.save(cache_label_file, label)
        # ^ This seems to take forever (but 5 mins at my laptop) so I want to investigate why and how we can improve.
        print(
            "Saving labels into cached file {} [took {:.3f} m]".format(cache_label_file, (time.time() - start)/60)
        )

      catlocs = tuple()
      for cat in range(self.n_class):
        catlocs += (np.argwhere(label == cat).reshape(-1),)
      
      self.catlocs_all_domain[domain] = catlocs

    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.deterministic = deterministic

    self.domain_features, self.domain_kdes = get_domain_features_domain_kdes()

    self.embeddings_target, self.kde_target = get_kde_target()
    # self.embeddings_target [1000, 768]
  
  def __len__(self):
    return self.n_batch * self.n_episode
  
  def __getitem__(self, index):
    max_attempts = 10  # or any number you deem suitable
    attempts = 0
    while attempts < max_attempts:
        retval = self._help_getitem(index)
        if retval:
            return retval
        attempts += 1
        index = (index + 10000)  # Try the next item. Wrap around if we reach the end.
    raise ValueError(f"Failed to get a valid item after {max_attempts} attempts.")

     

  def _help_getitem(self, index):
    if self.deterministic:
      np.random.seed(index)  ## add for control # of tasks and # of images
    if len(self.domains) > 1:
      domain = np.random.choice(self.domains, 1)[0]
    elif len(self.domains) == 1:
      domain = self.domains[0]
    else:
      raise NotImplementedError
    
    # embeddings for one domain
    print(domain, index)
    train_embeddings = self.domain_features[domain]
    kde_train = self.domain_kdes[domain]


    catlocs = self.catlocs_all_domain[domain]
    dataset = self.datasets[domain]
    
    s, q = self.n_shot, self.n_query
    sv, qv = 1, 1
    shot, query = tuple(), tuple()
    
    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    ratios = []
    batch_embeddings = []
    for c in cats:

      idx = np.random.choice(catlocs[c], sv * s + qv * q, replace=False)      # random choose n_shot*shot_view + n_query*query_view (1*1+15*1) images in each classes
      
      embedding = train_embeddings[idx]
      batch_embeddings.append(embedding)

      ratio = compute_density_ratios(embedding, self.kde_target, kde_train)
      ratios.append(ratio)
      
      
      s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
      c_shot = torch.stack([self.transform(dataset[i][0]) for i in s_idx])          # [5(1),3,84,84] [S*SV, C, H ,W]
      c_query = torch.stack([self.transform(dataset[i][0]) for i in q_idx])         # [15,3,84,84] [Q*QV, C, H ,W]
      c_shot = c_shot.view(sv, s, 1, *c_shot.shape[-3:])   # hard code V = 1             # [1,5(1),1,3,84,84] [SV, S, V, C, H ,W]
      c_query = c_query.view(qv, q, *c_query.shape[-3:])                           # [1,15,3,84,84] [QV, Q, C, H ,W]
      shot += (c_shot,)
      query += (c_query,)
    
    # diversity
    batch_embeddings = torch.concat(batch_embeddings)
    # print("batch_embeddings shape: ", batch_embeddings.shape) # [Y*(s+q), D] [150, 768]

    recons_error = compute_reconstruction_error(batch_embeddings, self.embeddings_target)
    # print("recons error: ", recons_error)
    
    ratios = np.concatenate(ratios).reshape(-1)
    if (ratios < 0.5).sum() > ratios.shape[0] // 2 or recons_error > 27.9:
      #  print("none!!!")
       return None

    shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]                   # [1, 5, 1, 3, 84, 84] 
    query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]                      # [1, 50, 3, 84, 84]
    cats = torch.from_numpy(cats)
    return shot, query, cats
  
def compute_reconstruction_error(A, x):
    x = x.t()
    if torch.cuda.is_available():
      x = x.float().cuda()
      A = A.float().cuda()
    else:
      x = x.float()
    # Compute the SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices = False)

    # Project x into the space defined by U
    VVTx = Vh.t() @ Vh @ x

    # Compute the reconstruction error
    error = torch.norm(VVTx - x)

    return error