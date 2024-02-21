import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['UnsupMoCoLoss', 'SupMoCoLoss', 'SimCLRLoss']


class ContrastLoss(nn.Module):
  def __init__(self, reduction='mean', T=0.07):
    super(ContrastLoss, self).__init__()

    self.reduction = reduction
    self.T = T  # softmax temperature

  @torch.no_grad()
  def _category_iou(self, qry_labels, key_labels):
    """ Quantifies task similarity using intersection-over-union (IOU) of 
    their respective category sets. """
    q, qc = qry_labels.size()
    k, kc = key_labels.size()

    qry_labels = qry_labels.view(q, 1, qc, 1)
    key_labels = key_labels.view(1, k, 1, kc)
    n_int = torch.eq(qry_labels, key_labels).sum(dim=(-2, -1)).float()
    n_union = kc + qc - n_int
    iou = n_int.div_(n_union)
    return iou

  def forward(self, *input):
    raise NotImplementedError


class UnsupMoCoLoss(ContrastLoss):
  """ Un-/self-supervised MoCo loss (He et.al., CVPR 20). """
  def __init__(self, reduction='mean', T=0.07, K=8192, key_dim=128):
    super(UnsupMoCoLoss, self).__init__(reduction, T)

    self.K = K  # queue size

    self.register_buffer('queue', F.normalize(torch.randn(key_dim, K)))
    self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

  @torch.no_grad()
  def _update_queue(self, keys):
    """ Enqueues and dequeues keys. """
    ptr = int(self.ptr)
    new_ptr = ptr + keys.size(0)
    if new_ptr < self.K:
      self.queue[:, ptr:new_ptr] = keys.T
    else:
      new_ptr -= self.K
      self.queue[:, ptr:] = keys[:self.K - ptr].T
      self.queue[:, :new_ptr] = keys[self.K - ptr:].T
    self.ptr[0] = new_ptr

  def forward(self, qrys, keys):
    assert qrys.dim() == 3            # [QV, B, D]
    assert keys.dim() == 3            # [KV, B, D]
    assert qrys.size(1) == keys.size(1)
    assert qrys.size(2) == keys.size(2)
    
    QV, B, _ = qrys.size()
    KV, _, _ = keys.size()

    # logits
    new_logits = torch.sum(
      qrys.unsqueeze(0) * keys.unsqueeze(1), dim=-1)        # [KV, QV, B]
    new_logits = new_logits.flatten(1)                      # [KV, QV * B]
    old_logits = torch.mm(
      qrys.flatten(0, 1), self.queue.clone())               # [QV * B, K]
    logits = torch.cat([new_logits.T, old_logits], dim=1)   # [QV * B, KV + K]
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - max_logits.detach() # for numerical stability
    logits = logits / self.T

    # similarity map
    new_sim = torch.ones_like(new_logits)
    old_sim = torch.zeros_like(old_logits)
    sim = torch.cat([new_sim.T, old_sim], dim=1)

    # queue
    keys = keys.flatten(0, 1)
    self._update_queue(keys)

    # loss
    exp_logits = logits.exp()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    loss = -(log_prob * sim).sum(dim=1) / sim.sum(dim=1)
    if self.reduction == 'mean':
      loss = loss.mean()
    else:
      loss = loss.sum()
    return loss


class SupMoCoLoss(ContrastLoss):
  """ Supervised MoCo loss (Adapted from Khosla et.al., arXiv 20). """
  def __init__(self, reduction='mean', T=0.07, K=8192, key_dim=128, label_dim=1):
    super(SupMoCoLoss, self).__init__(reduction, T)

    self.K = K  # queue size

    self.register_buffer('key_queue', F.normalize(torch.randn(key_dim, K)))
    self.register_buffer('label_queue', -torch.ones(K, label_dim))
    self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

  @torch.no_grad()
  def _update_queue(self, keys, labels):
    """ Enqueues and dequeues keys and their associated labels. """
    ptr = int(self.ptr)
    new_ptr = ptr + keys.size(0)
    if new_ptr < self.K:
      self.key_queue[:, ptr:new_ptr] = keys.T
      self.label_queue[ptr:new_ptr] = labels
    else:
      new_ptr -= self.K
      self.key_queue[:, ptr:] = keys[:self.K - ptr].T
      self.key_queue[:, :new_ptr] = keys[self.K - ptr:].T
      self.label_queue[ptr:] = labels[:self.K - ptr]
      self.label_queue[:new_ptr] = labels[self.K - ptr:]
    self.ptr[0] = new_ptr

  def forward(self, qrys, keys, labels):
    assert qrys.dim() == 3            # [QV, B, D]
    assert keys.dim() == 3            # [KV, B, D]
    assert qrys.size(1) == keys.size(1)
    assert qrys.size(2) == keys.size(2)
    QV, B, _ = qrys.size()
    KV, _, _ = keys.size()

    assert labels.size(0) == B
    if labels.dim() == 1:
      labels = labels.view(B, 1)
    assert labels.dim() == 2

    # queue
    keys = keys.flatten(0, 1)
    labels = labels.repeat(KV, 1)
    self._update_queue(keys, labels)

    # logits
    logits = torch.mm(qrys.flatten(0, 1), self.key_queue.clone())
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - max_logits.detach() # for numerical stability
    logits = logits / self.T

    # similarity map
    if labels.size(1) == 1:
      sim = torch.eq(labels.float(), self.label_queue.T).float()
    else:
      sim = self._category_iou(labels.float(), self.label_queue)
    sim = sim.repeat(QV, 1)

    # loss
    exp_logits = logits.exp()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    loss = -(log_prob * sim).sum(dim=1) / sim.sum(dim=1)
    if self.reduction == 'mean':
      loss = loss.mean()
    else:
      loss = loss.sum()
    return loss


class SimCLRLoss(ContrastLoss):
  """ SimCLR loss (Chen et.al., ICML 20 & Khosla et.al., arXiv 20). """
  def __init__(self, reduction='mean', T=0.07):
    super(SimCLRLoss, self).__init__(reduction, T)

  def forward(self, feats, labels=None):
    assert feats.dim() == 3           # [V, B, D]
    
    V, B, _ = feats.shape
    assert V > 1
    if labels is not None:
      assert labels.size(0) == B
      if labels.dim() == 1:
        labels = labels.view(B, 1)
      assert labels.dim() == 2

    feats = feats.flatten(0, 1)

    # logits
    logits = torch.mm(feats, feats.T)
    max_logits, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - max_logits.detach() # for numerical stability
    logits = logits / self.T
    
    # similarity map
    if labels is None:
      sim = torch.eye(B).to(feats.device) 
    else:
      if labels.size(1) == 1:
        sim = torch.eq(labels, labels.T).float()
      else:
        sim = self._category_iou(labels, labels)  
    sim = sim.repeat(V, V)
    mask = torch.ones_like(sim).scatter_( # avoids self-contrasting
      1, torch.arange(V * B).view(-1, 1).to(feats.device), 0)
    sim.mul_(mask)

    # loss
    exp_logits = logits.exp() * mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
    loss = -(log_prob * sim).sum(dim=1) / sim.sum(dim=1)
    if self.reduction == 'mean':
      loss = loss.mean()
    else:
      loss = loss.sum()
    return loss