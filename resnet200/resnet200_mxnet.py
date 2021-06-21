import os
import sys

import cv2
import mxnet as mx
import numpy as np


def change_device(arg_params, aux_params, ctx):
  new_args = dict()
  new_auxs = dict()
  for k, v in arg_params.items():
    new_args[k] = v.as_in_context(ctx)
  for k, v in aux_params.items():
    new_auxs[k] = v.as_in_context(ctx)
  return new_args, new_auxs


class Resnet200Extractor(object):
  def __init__(self, gpuid, batch_size, out_name='pool1_output'):
    self.ctx = mx.gpu(gpuid)
    self.batch_size = batch_size
    self.arg_params = None
    self.aux_params = None
    self.out = None
    self.exe = None
    self.h = -1
    self.w = -1

    self.out_name = out_name # 'relu1_output', 'pool1_output', 'fc1-conv_output'

  def load_model(self, model_prefix, model_epoch):
    symbol, self.arg_params, self.aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
    self.arg_params, self.aux_params = change_device(self.arg_params, self.aux_params, self.ctx)

    outs = symbol.get_internals()
    self.out = outs[self.out_name]

  def set_hw(self, h, w):
    self.h = h
    self.w = w
    self.arg_params['data'] = mx.nd.zeros((self.batch_size, 3, h, w), self.ctx)
    self.arg_params[self.out_name] = mx.nd.empty((1,), self.ctx)
    self.exe = self.out.bind(self.ctx, self.arg_params, args_grad=None, grad_req='null', aux_states=self.aux_params)

  # imgs: list of imgs
  def extract_feature(self, imgs, rgb_or_bgr=True):
    assert len(imgs) <= self.batch_size
    pad = self.batch_size - len(imgs)

    batch_data = np.zeros((self.batch_size, 3, self.h, self.w), dtype=np.float32)
    imgs = np.array(imgs, dtype=np.float32)
    if rgb_or_bgr: # if bgr, convert to rgb
      imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = np.moveaxis(imgs, [0, 1, 2, 3], [0, 2, 3, 1]) # (None, c, h, w)
    batch_data[:len(imgs), ::] = imgs
    self.arg_params['data'][::] = mx.nd.array(batch_data, self.ctx)

    self.exe.forward(is_train=False)

    out = self.exe.output_dict[self.out_name].asnumpy()
    out = np.moveaxis(out, [0, 1, 2, 3], [0, 3, 1, 2])
    return out if pad == 0 else out[:-pad]

  def preprocess_img(self, img=None, img_path=None, min_size=None, max_size=None):
    if img is None:
      img = cv2.imread(img_path)
    if img is None:
      return None
    h, w, c = img.shape

    if min_size is not None:
      if h < w:
        nh = min_size
        nw = w * min_size // h
      else:
        nw = min_size
        nh = h * min_size // w
    elif max_size is not None:
      if h > w:
        nh = max_size
        nw = w * max_size // h
      else:
        nw = max_size
        nh = h * max_size // w
    else:
      nh = h
      nw = w

    if min_size is not None or max_size is not None:
      img = cv2.resize(img, (nw, nh))
      
    return img
