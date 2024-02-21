import torch.optim as optim


def make(name, params, **kwargs):
  if name == 'sgd':
    optimizer = optim.SGD(
      params, lr=kwargs['lr'], 
      momentum=(kwargs.get('momentum') or 0.), 
      weight_decay=(kwargs.get('weight_decay') or 0.)
    )
  elif name == 'rmsprop':
    optimizer = optim.RMSprop(
      params, lr=kwargs['lr'], 
      momentum=(kwargs.get('momentum') or 0.), 
      weight_decay=(kwargs.get('weight_decay') or 0.)
    )
  elif name == 'adam':
    optimizer = optim.Adam(
      params, lr=kwargs['lr'], 
      betas=(kwargs.get('beta1') or 0.9, kwargs.get('beta2') or 0.999),
      weight_decay=(kwargs.get('weight_decay') or 0.)
    )
  elif name == 'adamW':
    optimizer = optim.AdamW(
      params, lr=kwargs['lr'], 
      betas=(kwargs.get('beta1') or 0.9, kwargs.get('beta2') or 0.999),
      weight_decay=(kwargs.get('weight_decay') or 0.)
    )
  else:
    raise ValueError('invalid optimizer: {}'.format(name))
  return optimizer


def load(ckpt, params):
  optimizer = make(ckpt['optimizer'], params, **ckpt['optimizer_args'])
  optimizer.load_state_dict(ckpt['optimizer_state_dict'])
  return optimizer