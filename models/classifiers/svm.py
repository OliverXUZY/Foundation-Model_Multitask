import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC

from .classifiers import register


@register('fs-svm')
class FSSVMClassifier(nn.Module):
  def __init__(self, in_dim=None, n_way=None, C=10.):
    super(FSSVMClassifier, self).__init__()

    self.clf = SVC(C=C, kernel='linear', random_state=0)

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
        out = self.clf.predict(q[0, j] if QV == 1 else q[i, j])
        pred = np.zeros((len(out), Y))
        pred[np.arange(len(out)), out] = 1.
        v_preds += (pred,)
      v_preds = np.stack(v_preds)
      preds += (v_preds,)
    preds = np.stack(preds)
    preds = torch.from_numpy(preds).cuda()
    return preds