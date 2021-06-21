import os
import argparse

import cv2
import mxnet as mx
import numpy as np

from resnet200_mxnet import Resnet200Extractor


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir')
  parser.add_argument('--min_size', type=int, default=None)
  parser.add_argument('--max_size', type=int, default=None)
  parser.add_argument('--ft_name', default='pool1_output')
  return parser

def extract_video_features():
  parser = build_parser()
  parser.add_argument('--video_dir')
  parser.add_argument('--name_file')
  parser.add_argument('--frame_gap', default=16, type=int)
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--start_id', type=int, default=0)
  parser.add_argument('--end_id', type=int, default=None)
  parser.add_argument('--output_dir')
  opts = parser.parse_args()

  model_prefix = os.path.join(opts.model_dir, 'fullconv-resnet-imagenet-200-0')
  model_epoch = 22
  gpuid = 0

  resnet200 = Resnet200Extractor(gpuid, opts.batch_size, out_name=opts.ft_name)
  resnet200.load_model(model_prefix, model_epoch)

  if not os.path.exists(opts.output_dir):
    os.makedirs(opts.output_dir)

  names = np.load(opts.name_file)
  if opts.end_id is None:
    opts.end_id = len(names)

  for i, name in enumerate(names[opts.start_id: opts.end_id]):
    video_path = os.path.join(opts.video_dir, name)
    output_path = os.path.join(opts.output_dir, name+'.npy')
    if os.path.exists(output_path):
      continue

    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    t = 0
    while success:
      frame = resnet200.preprocess_img(img=frame, 
        min_size=opts.min_size, max_size=opts.max_size)
      if t % opts.frame_gap == 0:
        frames.append(frame)
      success, frame = cap.read()
      t += 1
    if len(frames) == 0:
      print('empty video', name)
      continue
    h, w, c = frames[0].shape
    resnet200.set_hw(h, w)
    cap.release()

    fts = []
    for t in range(0, len(frames), opts.batch_size):
      ft = resnet200.extract_feature(frames[t: t+opts.batch_size], rgb_or_bgr=True)
      # global spatial pooling
      fts.extend(np.mean(ft.reshape((len(frames[t: t+opts.batch_size]), -1, ft.shape[-1])), axis=1))
    fts = np.array(fts, np.float32)
    with open(output_path, 'wb') as f:
      np.save(f, fts)
    
    if i % 10 == 0:
      print(name, len(frames), frames[0].shape, fts.shape)


if __name__ == '__main__':
  extract_video_features()
