import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression

from .classifiers import register


@register('logistic')
class LogisticClassifier(nn.Module):
  def __init__(self, in_dim, n_way):
    super(LogisticClassifier, self).__init__()

    self.clf = nn.Linear(in_dim, n_way)

  def forward(self, x):
    logits = self.clf(x)
    return logits


# @register('fs-logistic')
# class FSLogisticClassifier(nn.Module):
#   def __init__(self, in_dim, n_way):
#     super(FSLogisticClassifier, self).__init__()

#     self.clf = nn.Linear(in_dim, n_way)

#   def _reset(self):
#     nn.init.constant_(self.clf.weight, 0.)
#     nn.init.constant_(self.clf.bias, 0.)

#   def optimize(self, x, y):
#     self._reset()
#     optimizer = optim.SGD(self.parameters(), lr=0.25, momentum=0.9)
#     for _ in range(400):
#       def closure():
#         optimizer.zero_grad()
#         logits = self.clf(x)
#         loss = F.cross_entropy(logits, y)
#         loss.backward()
#         return loss
#       optimizer.step(closure)
#     optimizer.zero_grad()

#   def forward(self, s, q, normalize=True):
#     #############################################
#     # E: number of episodes / tasks
#     # SV: number of shot views per task
#     # QV: number of query views per task
#     # V: number of views per image
#     # Y: number of ways / categories per task
#     # S: number of shots per category
#     # Q: number of queries per category
#     # D: input / feature dimension
#     #############################################
#     assert s.dim() == 6                             # [SV, E, Y, S, V, D]
#     assert q.dim() == 4                             # [QV, E, Y * Q, D]
#     assert q.size(0) in [1, s.size(0)]

#     if normalize:
#       s = F.normalize(s, dim=-1)
#       q = F.normalize(q, dim=-1)

#     SV, E, Y, S, V = s.shape[:-1]
#     QV = q.shape[0]
#     s = s.view(SV, E, Y * S * V, -1)
    
#     y = torch.arange(Y)[:, None].repeat(1, S * V).flatten().cuda()
#     logits = tuple()
#     for i in range(SV):
#       v_logits = tuple()
#       for j in range(E):
#         with torch.enable_grad():
#           self.optimize(s[i, j], y)
#         out = self.clf(q[0, j] if QV == 1 else q[i, j])
#         v_logits += (out,)
#       v_logits = torch.stack(v_logits)
#       logits += (v_logits,)
#     logits = torch.stack(logits)
#     return logits


@register('fs-logistic')
class FSLogisticClassifier(nn.Module):
  def __init__(self, in_dim=None, n_way=None, C=10.):
    super(FSLogisticClassifier, self).__init__()

    self.clf = LogisticRegression(penalty='l2', C=C, solver='lbfgs', 
      max_iter=1000, multi_class='multinomial', random_state=0)

  def forward(self, s, q, normalize=True):
    #############################################
    # E: number of episodes / tasks
    # SV: number of shot views per task
    # QV: number of query views per task
    # V: number of views per image
    # Y: number of ways / categories per task
    # S: number of shots per category
    # Q: number of queries per category
    # D: input / feature dimension
    #############################################
    assert s.dim() == 6                             # [SV, E, Y, S, V, D]
    assert q.dim() == 4                             # [QV, E, Y * Q, D]
    assert q.size(0) in [1, s.size(0)]

    if normalize:
      s = F.normalize(s, dim=-1)
      q = F.normalize(q, dim=-1)

    SV, E, Y, S, V = s.shape[:-1]
    QV = q.shape[0]
    s = s.view(SV, E, Y * S * V, -1).cpu()
    q = q.cpu()
    
    y = np.arange(Y)[:, None].repeat(S * V)
    preds = tuple()
    for i in range(SV):
      v_preds = tuple()
      for j in range(E):
        f = self.clf.fit(s[i, j], y)
        pred = self.clf.predict_proba(q[0, j] if QV == 1 else q[i, j])     # [50, 5]  [Y * Q, Y]
        v_preds += (pred,)
      v_preds = np.stack(v_preds)                  # [4,50,5] [E, Y * Q, Y]
      preds += (v_preds,)
    preds = np.stack(preds)                        # [1, 4,50,5] [SV, E, Y * Q, Y]
    preds = torch.from_numpy(preds).cuda()
    return preds
