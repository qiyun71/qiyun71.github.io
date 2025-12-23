---
title: Template of project
date: 2025-08-01 12:45:46
tags: 
categories: Project
---

> [qiyun71/Template-of-project](https://github.com/qiyun71/Template-of-project)

Project directory：
- config：case.yaml
- data：case data dir
- dataset：case data preprocess
- network：NN code
- outputs：save dir of ckpt and config and logs
- system：train/test/val.... code
- utils: tools code
- run.py
- README.md


Case directory：
- 0Model
- 1DataGeneration
- 2Experiment

<!-- more -->


```bash
python run.py --config xxx.yaml --args xxx any.conf=yyy

eg:
python run.py --config nasa.yaml --stage train system.max_epochs=2000
```

## run.py

```python
import argparse
import torch
import importlib
# importlib.import_module(module, package=None)
# __import__(module,fromlist=[None])
from utils.io_utils import seed_everything
from utils.conf import load_config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument('--stage', default=None, help='train or test or exp')
    parser.add_argument('--seed', type=int, default=1771, help='seed for initializing training.')
    
    args, extras = parser.parse_known_args()
    return args, extras

if __name__ == "__main__":
    args, extras = get_args()

    if args.seed is not None:
        seed_everything(args.seed)

    config = load_config(args.config, cli_args=extras)

    # model
    model_config = config["model"]
    module, cls = model_config["func"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    model = cls(**model_config.get("params", dict()))

    # dataset
    dataset_config = config["dataset"]
    module, cls = dataset_config["func"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    dataset = cls(args.stage, **dataset_config.get("datasets", dict()))
    dataloader = torch.utils.data.DataLoader(dataset, **dataset_config.get(args.stage, dict()))

    # system
    system_config = config["system"]
    module, cls = system_config["func"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    Trainer = cls(model, dataloader, config, args)

    if args.stage == "train":
        Trainer.train()
    elif args.stage == "test":
        Trainer.test()
    elif args.stage == "exp":
        Trainer.exp()
```

## config

```yaml
model:
  func: network.SelfSupervisedModel.TimeSeriesSM
  params:
    in_dim: 9
    out_dim: 5001
    dim_hidden: 128 # 256
    n_hidden_layers: 4
    sample_interval: .001
    sample_duration: 5.0
    fft_freq_divide: 8

dataset:
  func: dataset.NASA_SubA_SM.Dataset
  datasets:
    root_dir: data/NASA_SubA/
    normalize: True
    normalize_file: data/NASA_SubA/train/aey.npz
  train:
    batch_size: 64
    shuffle: True
  test:
    batch_size: 50 # 2000
    shuffle: False
  exp:
    batch_size: 100
    shuffle: False

system:
  func: system.NASA_SubA_SM.Trainer
  save_dir: './outputs/SM/'
  max_epochs: 100 # max_step = max_epochs * train_size / batch_size
  save_step: 1000
  fft_loss_type: l1
  mae_weight: 1.0
  fft_weight: 15.0 # 15.0
  mse_weight: 0.0
  lr:
    base_lr: 5.0e-3
    constant_lr_iters: 500
    exp_lr_gamma: 0.9999
```


## data

case_name:
- train
- test
- exp

## dataset

### normalize

对输入参数/输出固有频率归一化时，假如有n个样本，只需要找到这n个样本的最大最小值即可归一化

对输出的序列数据(时序/频响)归一化时，