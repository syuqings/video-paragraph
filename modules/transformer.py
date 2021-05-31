import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import framework.configbase
import math
import time
import numpy as np
from modules.transformer_encoder import Encoder
from modules.transformer_decoder import Decoder

decay1 = [(i+1)*20**(-1) for i in range(20)]
decay2 = [1-(i+1)*50**(-1) for i in range(50)]


class TransformerConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super(TransformerConfig, self).__init__()
    self.vocab = 0
    self.max_words_in_sent = 150
    self.ft_dim = 4096
    self.d_model = 512
    self.enc_n_layers = 3
    self.dec_n_layers = 3
    self.heads = 8
    self.dropout = 0.1
    self.keyframes = False
    self.rl = False
    self.document_freq = None

    
class Transformer(nn.Module):
  def __init__(self, config):
    super(Transformer, self).__init__()
    self.config = config
    self.encoder = Encoder(self.config.ft_dim, self.config.d_model, self.config.enc_n_layers, self.config.heads, self.config.dropout, self.config.keyframes)
    self.decoder = Decoder(self.config.vocab, self.config.d_model, self.config.dec_n_layers, self.config.heads, self.config.dropout)
    self.dropout = nn.Dropout(self.config.dropout)
    self.logit = nn.Linear(self.config.d_model, self.config.vocab)
    self.logit.weight = self.decoder.embed.embed.weight
    self.remove_gate = nn.Linear(self.config.d_model, 1)
    self.add_gate = nn.Linear(self.config.d_model, 1)
    self.q_linear = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
    self.next_attn = nn.Linear(2*self.config.d_model, 1)
    self.init_weights()

  def init_weights(self,):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, src, trg, src_mask, trg_mask):
    e_outputs, org_key, select = self.encoder(src, src_mask)
    add_state = torch.tensor(decay2[:e_outputs.size(1)]+[0]*max(0,e_outputs.size(1)-50)).cuda().unsqueeze(0).unsqueeze(-1)
    memory_bank = e_outputs * add_state
    d_output, attn_weights = [], []
    for i in range(1, trg.size(1)+1):
      word, attn = self.decoder(trg[:,i-1].unsqueeze(1), memory_bank, src_mask, trg_mask[:,i-1,:i].unsqueeze(1), step=i)
      d_output.append(word[:,-1])
      attn_weights.append(attn[:,:,-1].mean(dim=1))
      memory_bank, add_state = self.update_memory(memory_bank, add_state, e_outputs, attn_weights[-20:], d_output[-20:])
    output = self.logit(torch.cat([_.unsqueeze(1) for _ in d_output], 1))
    return output, org_key, select

  def update_memory(self, memory_bank, add_state, e_outputs, attn, query_s):
    remove_prob = torch.sigmoid(self.remove_gate(query_s[-1])).unsqueeze(-1)
    add_prob = torch.sigmoid(self.add_gate(query_s[-1])).unsqueeze(-1)
    temp = torch.softmax(torch.tensor(decay1[20-len(attn):]).cuda(), dim=-1)
    attn = sum([attn[i]*temp[i] for i in range(len(attn))]).unsqueeze(-1)
    # remove for diversity
    query_s = sum([query_s[i]*temp[i] for i in range(len(query_s))])
    sim = torch.sigmoid(torch.matmul(memory_bank, self.q_linear(query_s).unsqueeze(-1)))
    memory_bank = memory_bank * (1 - remove_prob * attn * sim)
    # add for coherence
    last_ctx = (e_outputs * attn).sum(dim=1, keepdim=True)
    next_attn = torch.sigmoid(self.next_attn(torch.cat([e_outputs,last_ctx.expand_as(e_outputs)], dim=-1)))
    memory_bank = memory_bank + e_outputs * (1-add_state) * (add_prob*next_attn)
    add_state = add_state + (1-add_state) * (add_prob*next_attn)
    return memory_bank, add_state

  def sample(self, src, src_mask, decoding='greedy'):
    init_tok = 2
    eos_tok = 3
    if self.config.keyframes:
      e_outputs, src_mask = self.encoder.get_keyframes(src, src_mask)
    else:
      e_outputs, _, _ = self.encoder(src, src_mask)
    add_state = torch.tensor(decay2[:e_outputs.size(1)]+[0]*max(0,e_outputs.size(1)-50)).cuda().unsqueeze(0).unsqueeze(-1)
    memory_bank = e_outputs * add_state
    outputs = torch.ones(src.size(0), 1).fill_(init_tok).long().cuda()
    seqLogprobs = torch.zeros(src.size(0), 60).cuda()
    attn_weights, d_output = [], []
    for i in range(1, 60):
      trg_mask = self.nopeak_mask(i)
      word, attn = self.decoder(outputs[:,-1].unsqueeze(1), memory_bank, src_mask, trg_mask[:,-1].unsqueeze(1), step=i)
      attn_weights.append(attn[:,:,-1].mean(dim=1))
      d_output.append(word[:,-1])
      out = self.logit(word)
      logprobs = F.log_softmax(out[:,-1], dim=-1)
      if decoding == 'greedy':
        _, next_word = torch.max(logprobs, dim=1)
        next_word = next_word.unsqueeze(-1)
      else:
        probs = torch.exp(logprobs.data).cpu()
        next_word = torch.multinomial(probs, 1).cuda()
        seqLogprobs[:,i] = logprobs.gather(1, next_word).view(-1)
      outputs = torch.cat([outputs, next_word], dim=1)
      memory_bank, add_state = self.update_memory(memory_bank, add_state, e_outputs, attn_weights[-20:], d_output[-20:])
    attn_weights = torch.cat([_.unsqueeze(1) for _ in attn_weights], dim=1)
    return outputs, seqLogprobs, attn_weights

  def nopeak_mask(self, size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).cuda()
    return np_mask

