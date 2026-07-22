## Introduction

- VI-E2E is a V2I cooperative end-to-end planning framework that integrates infrastructure-side observation information using instance-level query representations for combined AD tasks.
- We design a planning-aware map-masking strategy that suppresses low-confidence or less planning-relevant map regions before trajectory generation.
- We introduce a temporal compensation module that uses predicted motion to offset infrastructure time lag and prevent stale features from degrading agent reasoning and planning.
- We propose a confidence-guided V2I query fusion module that performs class-consistent, distance-gated Hungarian matching between vehicle and infrastructure agent queries, followed by confidence-weighted anchor fusion and learned query-feature fusion.


## Environment Installation
Following https://github.com/hustvl/VAD/blob/main/docs/install.md

## Dataset Preparation
Download the Dair-v2x dataset and convert it into Nuscenes dataset format. Refer to the Dair-v2x official dataset conversion file, following https://github.com/AIR-THU/UniV2X/blob/main/docs/DATA_PREP.md. 

## Train and eval

## Train VAD with 8 GPUs 
```shell
cd /path/to/VI-E2E
conda activate vi-e2e
python -m torch.distributed.run --nproc_per_node=8 --master_port=2333 tools/train.py projects/configs/V2X_VAD/V2X_VAD_base_e2e_patch.py --launcher pytorch --deterministic --work-dir path/to/save/outputs
```

**NOTE**: We release two types of training configs: the end-to-end configs and the two-stage (stage-1: Perception, stage-2: Planning) configs. For stage 1, set the planning related parameters into 0 in the config file. After getting the reasonable perception results, then start training the planning modules.

## Eval VAD with 1 GPU
```shell
cd /path/to/VAD
conda activate vad
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/V2X_VAD/V2X_VAD_base_e2e_patch.py /path/to/ckpt.pth --launcher none --eval bbox --tmpdir tmp
```


## Reproduce results with pre-trained weights
If you want to reproduce results with pre-trained weights, please change the `img_norm_cfg` setting in your config file to following:

 ``` 
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
```

## Results
| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s |  |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: |
| No Fusion | 1.0832 | 1.8122 | 2.5887 | 0.0027 | 0.0059 | 0.0073 |
| UniV2X | 1.4329 | 2.1256 | 2.9520 | 0.0015 | 0.0015 | 0.0074 | 
| VI-E2E | **1.0087** | **1.6383** | **2.2712** | **0.0000** | **0.0005** | **0.0006** |




