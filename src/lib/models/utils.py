from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  # sigmoid 的时候限制数值避免溢出
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    """ 生成新特征 """
    dim  = feat.size(2) # 得到最后一层的特征个数为dim
    # ind 应该是 [batch,h*w]的，这里tile第三维 dim 。
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) 
    feat = feat.gather(1, ind) # 根据对应的index生成新的值。
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    """ 对特征进行转置和生成 """
    feat = feat.permute(0, 2, 3, 1).contiguous() # 估计是把bchw变成[b,h,w,c]，contiguous就是复制一个copy出来。
    feat = feat.view(feat.size(0), -1, feat.size(3)) # resize到 [b,h*w,c]
    feat = _gather_feat(feat, ind) # 根据index找到生成新特征
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)