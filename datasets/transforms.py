import random

import torch
import torchvision.transforms as transforms
from PIL import ImageFilter

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class MultiViewTransform(object):
  def __init__(self, transform, n_view=2):
    self.transform = transform
    self.n_view = n_view

  def __call__(self, x):
    views = torch.stack([self.transform(x) for _ in range(self.n_view)])
    return views


class GaussianBlur(object):
  def __init__(self, sigma=(.1, 2.)):
    self.sigma = sigma

  def __call__(self, x):
    sigma = random.uniform(self.sigma[0], self.sigma[1])
    x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
    return x


def get_transform(name, size, statistics=None):
  if statistics is None:
    statistics = {'mean': [0., 0., 0.],
                  'std':  [1., 1., 1.]}
                  
  if name in ['ucb', 'ucb-fs']:
    return transforms.Compose([
      transforms.RandomResizedCrop(size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'mit':
    return transforms.Compose([
      transforms.RandomCrop(size, padding=(8 if size > 32 else 4)),
      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'mit-fs':
    return transforms.Compose([
      transforms.RandomCrop(size, padding=(8 if size > 32 else 4)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'contrast':
    return transforms.Compose([
      transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([
        transforms.ColorJitter(
          brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
      ], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      # transforms.RandomApply([GaussianBlur()], p=0.5),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'contrast-fs':
    return transforms.Compose([
      transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'flip':
    return transforms.Compose([
      transforms.Resize(size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'enlarge':
    return transforms.Compose([
      transforms.Resize(int(size * 256 / 224)),
      transforms.CenterCrop(size),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'clip':
    return transforms.Compose([
      transforms.Resize(size, interpolation=BICUBIC),
      transforms.CenterCrop(size),
      _convert_image_to_rgb,
      transforms.ToTensor(),
      transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
  elif name == 'dinov2':
    # roughly copy from dinov2/data/augmentations.py
    return transforms.Compose([
      # color distorsions / blurring
      transforms.RandomResizedCrop(
          224, scale=(0.32, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
      ),
      transforms.RandomHorizontalFlip(p=0.5),
      # color distorsions / blurring
      transforms.RandomApply(
          [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
          p=0.8,
      ),
      transforms.RandomGrayscale(p=0.2),
      # normalization
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'make_classification_eval_transform':
    # roughly copy from dinov2/data/trainsforms.py L78
    # This matches (roughly) torchvision's preset for classification evaluation:
    #  https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
    return transforms.Compose([
      transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'mocov3':
    # roughly copied from mocov3/main_moco.py line 262:  https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L262
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**statistics)
    ]
    return transforms.Compose(augmentation1)
  elif name is None:
    return transforms.Compose([
      transforms.Resize(size),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  else:
    raise ValueError('invalid transform: {}'.format(name))

def _convert_image_to_rgb(image):
    return image.convert("RGB")