import torch
import torch.nn.functional as F
import torch.distributions as ds
import torch.autograd as autograd
import time
from .modules import Module


class Model(Module):
  def __init__(self, enc, head):
    super(Model, self).__init__()
    
    self.enc = enc    # encoder
    self.head = head  # classifier / projector
    # self.head = self.head.half()

  def _std_forward(self, x):
    assert x.dim() == 4                             # [B, C, H, W]
    # print(x.dtype)
    x = self.enc(x)
    # print("model x : ", x.dtype)
    # print("model: ", self.head.clf.weight.dtype)
    x = x.float()
    logits = self.head(x)
    return logits

  ####################################################
  # E: number of episodes / tasks per mini-batch
  # SV: number of shot views per task
  # QV: number of query views per task
  # V: number of views per image
  # Y: number of ways / categories per task
  # S: number of shots per category
  # Q: number of queries per category
  # M: number of input channels
  # N: number of filters / output channels
  # I: input spatial resolution
  # O: output spatial resolution
  # F: filter size (M x 3 x 3)
  # L: number of sliding windows (O x O)
  # T: total number of filters across layers
  # C: input channels
  # H: height of feature maps
  # W: width of feature maps
  # D: output / feature dimension
  ####################################################

  def _task2vec(self, h_list, gz_list, lead_dim, method='fim'):
    """ Calculates task2vec embeddings (Achille et.al., ICCV 19). """
    vecs = tuple()
    for h, gz in zip(h_list, gz_list):
      h_shape = h.shape[-3:]                        # [M, I, I]
      gz_shape = gz.shape[-3:]                      # [N, O, O]
      stride = h_shape[-1] // gz_shape[-1]          # assume full padding
      h = F.unfold(h, 3, padding=1, stride=stride)  # assume 3 x 3 kernels
      h = h.view(*lead_dim, *h.shape[-2:])          # [E, S, F, L]
      gz = gz.view(*lead_dim, *gz_shape)            # [E, S, N, O, O]
      gz = gz.flatten(-2)                           # [E, S, N, L]
      assert h.size(-1) == gz.size(-1)
      gz = gz.transpose(-2, -1)                     # [E, S, L, N]
      gw = torch.matmul(h, gz)                      # [E, S, F, N]
      if method == 'fim':
        v = gw.pow(2).mean(dim=(1, 2))              # [E, N]
      elif method == 'jac':
        v = gw.mean(dim=1).pow(2).mean(dim=1)       # [E, N]
      else:
        raise ValueError('invalid task2vec method: {}'.format(method))
      vecs += (v,) 
    vecs = torch.cat(vecs, dim=1)                   # [E, T]
    return vecs

  ####################################################
  # C: input channels
  # H: height of feature maps
  # W: width of feature maps
  # D: output / feature dimension
  ####################################################

  def _fs_forward(self, s, q, task2vec):
    assert s.dim() == 8                             # [E, SV, Y, S, V, C, H, W]
    assert q.dim() == 6                             # [E, QV, Y * Q, C, H, W]
    assert s.size(0) == q.size(0)
    assert q.size(1) in [1, s.size(1)]
    s = s.transpose(0, 1)                           # [SV, E, Y, S, V, C, H, W]
    q = q.transpose(0, 1)                           # [QV, E, Y * Q, C, H, W]

    SV, E, Y, S, V = s.shape[:-3]
    QV, _, YQ = q.shape[:-3]
    YSV, Q = Y * S * V, YQ // Y

    # timer= time.time()

    if task2vec:
      s = s.flatten(1, -4)                          # [SV, E * Y * S * V, C, H, W]
      q = q.flatten(1, -4)                          # [QV, E * Y * Q, C, H, W]
      x = torch.unbind(s) + torch.unbind(q)
      x, h, z = self.enc(x, hz=True)
      h, z = [h[-1]], [z[-1]]                       # only last layer for now
    else:
      s = s.flatten(0, -4)                          # [SV * E * Y * S * V, C, H, W]
      q = q.flatten(0, -4)                          # [QV * E * Y * Q, C, H, W]
      x = torch.cat([s, q])
      x = self.enc(x)
    # print("vision encoder done!, {} s".format(time.time() - timer))
    # timer= time.time()
    
    s, q = x[:SV*E*YSV], x[-QV*E*YQ:]
    s = s.view(SV, E, Y, S, V, -1)                  # [SV, E, Y, S, V, D]
    q = q.view(QV, E, YQ, -1)                       # [QV, E, Y * Q, D]
    
    logits = self.head(s, q)                        # [SV, E, Y * Q, Y]
    # print("vision clf done!, {} s".format(time.time() - timer))

    # computes task embeddings
    vecs = None
    if task2vec:
      vecs = tuple()
      lt = logits.flatten(1, 2)                     # [SV, E * Y * Q, Y]
      if task2vec == 'fim':
        y = ds.Categorical(logits=lt).sample()      # [SV, E * Y * Q]
        for i in range(SV):
          logp = -F.cross_entropy(lt[i], y[i], reduction='sum')
          if QV == 1:
            qh, qz = [k[-1] for k in h], [k[-1] for k in z]
          else:
            qh, qz = [k[-QV + i] for k in h], [k[-QV + i] for k in z]
          gqz = autograd.grad(logp, qz, create_graph=True)
          v = self._task2vec(qh, gqz, [E, YQ], task2vec)
          vecs += (v,)
      elif task2vec == 'jac':
        y = torch.arange(Y).repeat(Q, E).T
        y = y.flatten().cuda()                      # [E * Y * Q]
        for i in range(SV):
          loss = F.cross_entropy(lt[i], y, reduction='sum')
          sh, sz = [k[i] for k in h], [k[i] for k in z]
          if QV == 1:
            qh, qz = [k[-1] for k in h], [k[-1] for k in z]
          else:
            qh, qz = [k[-QV + i] for k in h], [k[-QV + i] for k in z]
          gz = autograd.grad(loss, sz + qz, create_graph=True)
          gsz, gqz = gz[:len(sz)], gz[-len(qz):]
          gsqz = [torch.cat(
                   [k1.view(E, YSV, *k1.shape[-3:]), 
                    k2.view(E, YQ, *k2.shape[-3:])], 
                    dim=1).flatten(0, 1) for k1, k2 in zip(gsz, gqz)]
          sqh = [torch.cat(
                   [k1.view(E, YSV, *k1.shape[-3:]), 
                    k2.view(E, YQ, *k2.shape[-3:])], 
                    dim=1).flatten(0, 1) for k1, k2 in zip(sh, qh)]
          v = self._task2vec(sqh, gsqz, [E, YSV+YQ], task2vec)
          vecs += (v,)
      else:
        raise ValueError('invalid task embedding: {}'.format(task2vec))
      vecs = torch.stack(vecs, dim=1)               # [E, SV, T]

    logits = logits.transpose(0, 1)                 # [E, SV, Y * Q, Y]
    return logits, vecs

  def forward(self, x, q=None, task2vec=None):
    if q is None: 
      return self._std_forward(x)
    else:
      return self._fs_forward(x, q, task2vec)