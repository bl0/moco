Unofficial implementation for [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

This repository carefully implemented important details such as ShuffleBN and distributed Queue mentioned in the paper to reproduce the reported results.

## Requirements

The following enverionments is tested:

* `Anaconda` with `python >= 3.6`
* `pytorch>=1.3, torchvision, cuda=9.2`
* `tensorboard_logger`: `pip install tensorboard_logger`

## Training Momentum Contrast on ImageNet

* The pre-training stage:

  ```
  exp_name=MoCo/ddp/8-gpu_bs-256_shuffle_bn
  python -m torch.distributed.launch --nproc_per_node=8 \
      train.py \
      --batch-size 32 \
      --exp-name ${exp_name} \
      --data-root /root/directory/of/datasets
  ```

  The checkpoints and tensorboard log will be saved in `./output/imagenet/${exp_name}`. Run `python train.py --help` for more help.
  
* The linear evaluation stage:

  ```
  exp_name=MoCo/ddp/8-gpu_bs-256_shuffle_bn
  python -m torch.distributed.launch --nproc_per_node=4 \
      eval.py \
      --exp-name ${exp_name} \
      --model-path ./output/imagenet/${exp_name}/models/current.pth \
      --batch-size 64 \
      --data-root /root/directory/of/datasets
  ```

  The checkpoints and tensorboard log will be saved in `./output/imagenet/${exp_name}`. Run `python eval.py --help` for more help.

## Pre-trained weights

Pre-trained model checkpoint and tensorboard log for K = 16384 and 65536 can be downloaded from [OneDrive](https://1drv.ms/u/s!AsaPPmtCAq08pEsUojFnhhnGLG8F?e=zFwbGY).

## Performance comparison

| K     | Acc@1 (ours) | Acc@1 (MoCo paper) |
| ----- | ------------ | ------------------ |
| 16384 | 59.89 ([model](https://1drv.ms/u/s!AsaPPmtCAq08pFfk01K2l2T7Hv9P?e=uI1vGx))        | 60.4               |
| 65536 | 60.79 ([model](https://1drv.ms/u/s!AsaPPmtCAq08pFa2xJRkILatNLh8?e=IMt2xg))       | 60.6               |

## Acknowledgements

A lot of codes is borrowed from [CMC](https://github.com/HobbitLong/CMC) and [lemniscate](https://github.com/zhirongw/lemniscate.pytorch).

