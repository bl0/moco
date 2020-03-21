Unofficial implementation for [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

## Highlight

1. **Effective**. Carefully implement important details such as ShuffleBN and distributed Queue mentioned in the paper to reproduce the reported results.
2. **Efficient**. The implementation is based on pytorch DistributedDataParallel and Apex automatic mixed precision. It only takes about 40 hours to train MoCo on imagenet dataset with 8 V100 gpus. The time cost is smaller than 3 days reported in original MoCo paper.


## Requirements

The following enverionments is tested:

* `Anaconda` with `python >= 3.6`
* `pytorch>=1.3, torchvision, cuda=10.1/9.2`
* others: `pip install termcolor opencv-python tensorboard`
* [Optional] [`apex`](https://github.com/NVIDIA/apex#quick-start): automatic mixed precision training.

## Train and eval on imagenet

* The pre-training stage:

  ```bash
  data_dir="./data/imagenet100"
  output_dir="./output/imagenet/K65536"
  python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
      train.py \
      --data-dir ${data_dir} \
      --dataset imagenet \
      --nce-k 65536 \
      --output-dir ${output_dir}
  ```

  The log, checkpoints and tensorboard events will be saved in `${output_dir}`. Set `--amp-opt-level` to `O1`, `O2`, or `O3` for mixed precision training. Run `python train.py --help` for more help.
  
* The linear evaluation stage:

  ```bash
  python -m torch.distributed.launch --nproc_per_node=4 \
      eval.py \
      --dataset imagenet \
      --data-dir ${data_dir} \
      --pretrained-model ${output_dir}/current.pth \
      --output-dir ${output_dir}/eval
  ```

  The checkpoints and tensorboard log will be saved in `${output_dir}/eval`. Set `--amp-opt-level` to `O1`, `O2`, or `O3` for mixed precision training. Run `python eval.py --help` for more help.


## Pre-trained weights

Pre-trained model checkpoint and tensorboard log for K = 16384 and 65536 on imagenet dataset can be downloaded from [OneDrive](https://1drv.ms/u/s!AsaPPmtCAq08pEsUojFnhhnGLG8F?e=zFwbGY).

BTW, the hyperparameters is also stored in model checkpoint, you can get full configs in the checkpoints like this:
```python
import torch
ckpt = torch.load('model.pth')
ckpt['opt']
```

## Performance comparison with original paper

| K     | Acc@1 (ours)                                                               | Acc@1 (MoCo paper) |
| ----- | -------------------------------------------------------------------------- | ------------------ |
| 16384 | 59.89 ([model](https://1drv.ms/u/s!AsaPPmtCAq08pFfk01K2l2T7Hv9P?e=uI1vGx)) | 60.4               |
| 65536 | 60.79 ([model](https://1drv.ms/u/s!AsaPPmtCAq08pFa2xJRkILatNLh8?e=IMt2xg)) | 60.6               |

## Notes

The MultiStepLR of pytorch1.4 is broken (See https://github.com/pytorch/pytorch/issues/33229 for more details). So if you are using pytorch1.4, you should not set `--lr-scheduler` to step. You can use `cosine` instead.

## Acknowledgements

A lot of codes is borrowed from [CMC](https://github.com/HobbitLong/CMC) and [lemniscate](https://github.com/zhirongw/lemniscate.pytorch).

