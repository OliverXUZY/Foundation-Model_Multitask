import torch


models = {}

def register(name):
  def decorator(cls):
    models[name] = cls
    return cls
  return decorator


def make(name, **kwargs):
  # for key, val in models.items():
    # print(key)
    # print(val)
    # print("zhuyan: ")
  if name is None:
    return None
  try:
    enc = models[name](**kwargs)
  except:
    raise ValueError('unsupported encoder: {}'.format(name))
  if torch.cuda.is_available():
    enc.cuda()
  return enc


def load(ckpt):
  enc = make(ckpt['encoder'], **ckpt['encoder_args'])
  if enc is not None:
    enc.load_state_dict(ckpt['encoder_state_dict'])
  return enc