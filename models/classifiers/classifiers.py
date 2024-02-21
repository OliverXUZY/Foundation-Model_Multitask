import torch


__all__ = ['make', 'load']


models = {}

def register(name):
  def decorator(cls):
    models[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  if name is None:
    return None
  try:
    clf = models[name](**kwargs)
  except:
    raise ValueError('unsupported classifier: {}'.format(name))
  if torch.cuda.is_available():
    clf.cuda()
  return clf


def load(ckpt):
  clf = make(ckpt['classifier'], **ckpt['classifier_args'])
  if clf is not None:
    clf.load_state_dict(ckpt['classifier_state_dict'])
  return clf