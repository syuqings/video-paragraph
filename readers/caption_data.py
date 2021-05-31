from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import random
import math
import torch.utils.data

UNK, PAD, BOS, EOS = 0, 1, 2, 3


class CaptionDataset(torch.utils.data.Dataset):
  def __init__(self, name_file, ft_root, cap_file, word2int, int2word,
    max_words_in_sent=150, is_train=False, _logger=None):
    super(CaptionDataset, self).__init__()

    if _logger is None:
      self.print_fn = print
    else:
      self.print_fn = _logger.info

    self.names = np.load(name_file)
    self.num_ft = len(self.names)
    self.print_fn('names size %d' % self.num_ft)

    self.ref_captions = json.load(open(cap_file))
    self.captions, self.cap2ftid = [], []
    for ftid, name in enumerate(self.names):
      self.captions.extend(self.ref_captions[name])
      self.cap2ftid.extend([ftid] * len(self.ref_captions[name]))
    self.cap2ftid = np.array(self.cap2ftid)
    self.num_caption = len(self.captions)
    self.print_fn('captions size %d' % self.num_caption)
    
    self.stoi = json.load(open(word2int))
    self.itos = json.load(open(int2word))
    self.ft_root = ft_root
    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train

  def temporal_pad_or_trim_feature(self, ft, max_len, transpose=False, average=False):
    length, dim_ft = ft.shape
    # pad
    if length <= max_len:
      ft_new = np.zeros((max_len, dim_ft), np.float32)
      ft_new[:length] = ft
    # trim
    else:
      if average:
        indices = np.round(np.linspace(0, length, max_len+1)).astype(np.int32)
        ft_new = [np.mean(ft[indices[i]: indices[i+1]], axis=0) for i in range(max_len)]
        ft_new = np.array(ft_new, np.float32)
      else:
        indices = np.round(np.linspace(0, length - 1, max_len)).astype(np.int32)
        ft_new = ft[indices]
    if transpose:
      ft_new = ft_new.transpose()
    return ft_new

  def pad_sent(self, x):
    max_len = self.max_words_in_sent
    padded = [BOS] + x[:max_len] + [EOS] + [PAD] * max(0, max_len - len(x))
    length = 1+min(len(x), max_len)+1
    return np.array(padded), length

  def sent2int(self, str_sent):
    int_sent = [self.stoi.get(w, UNK) for w in str_sent.split()]
    return int_sent

  def int2sent(self, batch):
    with torch.cuda.device_of(batch):
      batch = batch.tolist()
    batch = [[self.itos.get(str(ind), '<unk>') for ind in ex] for ex in batch] # denumericalize
    
    def trim(s, t):
      sentence = []
      for w in s:
        if w == t:
          break
        sentence.append(w)
      return sentence
    batch = [trim(ex, '<eos>') for ex in batch] # trim past frst eos

    def filter_special(tok):
      return tok not in ('<sos>', '<pad>')
    batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
    return batch

  def __len__(self):
    if self.is_train:
      return len(self.captions)
    else:
      return len(self.names)

  def __getitem__(self, idx):
    outs = {}
    if self.is_train:
      name = self.names[self.cap2ftid[idx]]
    else:
      name = self.names[idx]
    ft = []
    ft.append(np.load(os.path.join(self.ft_root, 'resnet200', '%s.mp4.npy'%name)))
    ft.append(np.load(os.path.join(self.ft_root, 'i3d.rgb', '%s.mp4.npy'%name)))
    try:
      ft.append(np.load(os.path.join(self.ft_root, 'i3d.flow', '%s.mp4.npy'%name)))
    except:
      pass
    ft = np.concatenate(ft, axis=-1)
    ft_len = min(150, len(ft))
    ft = self.temporal_pad_or_trim_feature(ft, 150)
    outs['ft_len'] = ft_len
    outs['img_ft'] = ft
    outs['name'] = name

    if self.is_train:
      outs['ref_sents'] = self.captions[idx]
      sent_id, sent_len = self.pad_sent(self.sent2int(self.captions[idx]))
      outs['caption_ids'] = sent_id
      outs['id_len'] = sent_len
    return outs
     

