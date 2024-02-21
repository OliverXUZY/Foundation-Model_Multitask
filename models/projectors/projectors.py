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
    prj = models[name](**kwargs)
  except:
    raise ValueError('unsupported projector: {}'.format(name))
  if torch.cuda.is_available():
    prj.cuda()
  return prj


def load(ckpt):
  prj = make(ckpt['projector'], **ckpt['projector_args'])
  if prj is not None:
    prj.load_state_dict(ckpt['projector_state_dict'])
  return prj