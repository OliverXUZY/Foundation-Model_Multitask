import os
import shutil
import time
import math

import torch
import numpy as np
import scipy.stats as stats


_log_path = None

def set_gpu(gpu):
  print('set gpu:', gpu)
  os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def check_path(path):
  if not os.path.exists(path):
    raise ValueError('path does not exist.')


def ensure_path(path, remove=True):
  basename = os.path.basename(path.rstrip('/'))
  if os.path.exists(path):
    # if remove and (basename.startswith('_')
    #   or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
    #   shutil.rmtree(path)
    #   os.makedirs(path)
    shutil.rmtree(path)
    os.makedirs(path)
  else:
    os.makedirs(path)

def make_path(path):
  if not os.path.exists(path):
    os.makedirs(path)


def set_log_path(path):
  global _log_path
  _log_path = path


def log(obj, filename='log.txt'):
  print(obj)
  if _log_path is not None:
    with open(os.path.join(_log_path, filename), 'a') as f:
      print(obj, file=f)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def item(self):
    return self.avg


class Timer(object):
  def __init__(self):
    self.start()

  def start(self):
    self.v = time.time()

  def end(self):
    return time.time() - self.v


def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600) # time in hours
  if t >= 60:
    return '{:.1f}m'.format(t / 60)   # time in minutes
  return '{:.1f}s'.format(t)


def warmup(low, high, epoch, n_epochs, batch, n_batches):
  """ Warm-up by linearly interpolation. """
  assert epoch <= n_epochs, 'already passed the warm-up phase.'
  assert low <= high
  p = (batch + (epoch - 1) * n_batches) / (n_epochs * n_batches)
  v = low + p * (high - low)
  return v


def decay_lr(epoch, n_epochs, **kwargs):
  """ Learning rate decay. """
  assert epoch <= n_epochs, 'already passed max epoch.'
  if kwargs.get('schedule') is None:
    lr = kwargs['lr']
  elif kwargs['schedule'] == 'cosine':
    decay_rate = kwargs.get('decay_rate') or 0.1
    eta_min = kwargs['lr'] * decay_rate ** 3
    lr = eta_min + (kwargs['lr'] - eta_min) * (
      1. + math.cos(math.pi * epoch / n_epochs)) / 2
  elif kwargs['schedule'] == 'step':
    decay_rate = kwargs.get('decay_rate') or 0.1
    decay_epochs = kwargs.get('decay_epochs') or []
    steps = np.sum(epoch > np.asarray(kwargs['decay_epochs']))
    lr = kwargs['lr'] * decay_rate ** steps
  else:
    raise ValueError(
      'invalid learning-rate schedule: {}'.format('schedule'))
  return lr


def accuracy(logits, labels, topk=(1,)):
  with torch.no_grad():
    batch_size = logits.size(0)
    assert batch_size == labels.size(0)

    _, pred = logits.topk(max(topk), 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    acc = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(dim=0)
      acc.append(correct_k * 100. / batch_size)
    return acc


def count_params(model, return_str=True):
  n_params = 0
  for p in model.parameters():
    n_params += p.numel()
  if return_str:
    if n_params >= 1e6:
      return '{:.1f}M'.format(n_params / 1e6)
    else:
      return '{:.1f}K'.format(n_params / 1e3)
  else:
    return n_params


def mean_confidence_interval(data, confidence=0.95):
  a = 1.0 * np.array(data)
  stderr = stats.sem(a)
  h = stderr * stats.t.ppf((1 + confidence) / 2., len(a) - 1)
  return h

## for finetune and train head  on pre-trained RN50 in CLIP
def set_bn_eval(module):
  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
    module.eval()