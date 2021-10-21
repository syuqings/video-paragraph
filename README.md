# Towards Diverse Paragraph Captioning for Untrimmed Videos
This repository contains PyTorch implementation of our paper [Towards Diverse Paragraph Captioning for Untrimmed Videos](https://arxiv.org/pdf/2105.14477.pdf) (CVPR 2021).

## Requirements
- Python 3.6
- Java 15.0.2
- PyTorch 1.2
- numpy, tqdm, h5py, scipy, six

## Training & Inference

### Data preparation
1. Download the pre-extracted video features of ActivityNet Captions or Charades Captions datasets from [BaiduNetdisk](https://pan.baidu.com/s/1NdlziFgGgSM__hOQi5mNKQ) (code: he21).
2. Decompress the downloaded files to the corresponding dataset folder in the ordered_feature/ directory.

### Start training
1. Train our model without reinforcement learning, ```*``` can be activitynet or charades.
```bash
$ cd driver
$ CUDA_VISIBLE_DEVICES=0 python transformer.py ../results/*/dm.token/model.json ../results/*/dm.token/path.json --is_train
```
&emsp;If you want to train the model with key frames selection, you can perform the following instruction instead.
```bash
$ cd driver
$ CUDA_VISIBLE_DEVICES=0 python transformer.py ../results/*/key_frames/model.json ../results/*/key_frames/path.json --is_train --resume_file ../results/*/key_frames/pretrained.th
```
&emsp;It will achieve a slightly worse result with only a half of the video features used at inference phase for faster decoding. You need to download the [pretrained.th](https://drive.google.com/file/d/1FdtYnrAv5dAuikOZLOiEvMBehFbY2CTz/view?usp=sharing) model at first for the key-frame selection.

2. Fine-tune the pretrained model in step 1 with reinforcement learning.
```bash
$ cd driver
$ CUDA_VISIBLE_DEVICES=0 python transformer.py ../results/*/dm.token.rl/model.json ../results/*/dm.token.rl/path.json --is_train --resume_file ../results/*/dm.token/model/epoch.*.th
```

### Evaluation
The trained checkpoints have been saved at the ```results/*/folder/model/``` directory. After evaluation, the generated captions (corresponding to the name file in the public_split) and evaluating scores will be saved at ```results/*/folder/pred/tst/```.
```bash
$ cd driver
$ CUDA_VISIBLE_DEVICES=0 python transformer.py ../results/*/folder/model.json ../results/*/folder/path.json --eval_set tst --resume_file ../results/*/folder/model/epoch.*.th
```
We also provide the pretrained models for the ActivityNet dataset [here](https://drive.google.com/file/d/1lROybafncTHOaleFw6h2ReHrI-ao98hx/view?usp=sharing) and Charades dataset [here](https://drive.google.com/file/d/1nrCRZsW4cRaLjNhCa9n0bXRpDe9hVJrx/view?usp=sharing), which are re-run and achieve similar results with the paper.

## Reference
If you find this repo helpful, please consider citing:
```
@inproceedings{song2021paragraph,
  title={Towards Diverse Paragraph Captioning for Untrimmed Videos},
  author={Song, Yuqing and Chen, Shizhe and Jin, Qin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```






