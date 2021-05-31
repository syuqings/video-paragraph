import torch
import torch.nn as nn
from modules.common import *
import time


class Embedder(nn.Module):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embed = nn.Embedding(vocab_size, d_model)
  def forward(self, x):
    return self.embed(x)


class DecoderLayer(nn.Module):
  def __init__(self, d_model, heads, dropout=0.1):
    super().__init__()
    self.norm_1 = Norm(d_model)
    self.norm_2 = Norm(d_model)
    self.norm_3 = Norm(d_model)
    self.dropout_1 = nn.Dropout(dropout)
    self.dropout_2 = nn.Dropout(dropout)
    self.dropout_3 = nn.Dropout(dropout)
    self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
    self.ff = FeedForward(d_model, dropout=dropout)

  def forward(self, x, e_outputs, src_mask, trg_mask, layer_cache=None):
    x2 = self.norm_1(x)
    x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask, layer_cache=layer_cache, attn_type='self')[0])
    x2 = self.norm_2(x)
    context, attn = self.attn_2(x2, e_outputs, e_outputs, src_mask, attn_type='context')
    x = x + self.dropout_2(context)
    x2 = self.norm_3(x)
    x = x + self.dropout_3(self.ff(x2))
    return x, attn
    
    
class Decoder(nn.Module):
  def __init__(self, vocab_size, d_model, N, heads, dropout):
    super().__init__()
    self.N = N
    self.embed = Embedder(vocab_size, d_model)
    self.pe = PositionalEncoder(d_model, dropout=dropout)
    self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
    self.norm = Norm(d_model)
    self.cache = None
  
  def _init_cache(self):
    self.cache = {}
    for i in range(self.N):
      self.cache['layer_%d'%i] = {
        'self_keys': None,
        'self_values': None,
      }    

  def forward(self, trg, e_outputs, src_mask, trg_mask, step=None):
    if step == 1:
      self._init_cache()

    x = self.embed(trg)
    x = self.pe(x, step)
    attn_w = []
    for i in range(self.N):
      layer_cache = self.cache['layer_%d'%i] if step is not None else None
      x, attn = self.layers[i](x, e_outputs, src_mask, trg_mask, layer_cache=layer_cache)
      attn_w.append(attn)
    return self.norm(x), sum(attn_w)/self.N
