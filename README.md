# Solution for CCKS2022 Track2

## 🌟 Introduction
This is the third place solution for the [2nd Task](https://tianchi.aliyun.com/competition/entrance/531956/introduction) of the CCKS-2022 Digital Business Knowledge Map Assessment Competition. 

📃Paper: "[Multi-Modal Representation Learning with Self-Adaptive Thresholds for Commodity Verification](https://arxiv.org/abs/2208.11064)"

<p align="center">
    <img width="50%" alt="model" src="https://github.com/hanchenchen/CCKS2022-track2-solution/files/9416232/figure1.pdf">
    <img width="30%" alt="similarity-pos-neg-sat" src="https://github.com/hanchenchen/CCKS2022-track2-solution/files/9794026/similarity-pos-neg-63.pdf">
</p>


### About Training Data

- The training is only conducted on the official training set. **Neither** external training data **nor** test data are utilized.
- When dividing the validation set, we remove the items that appear in the training set to ensure that the training set and validation set do not overlap. The ratio of the final training set and validation set is about 5.6:1.

### About Data Preprocessing

- We resize all images to 384 x 384.
- For text, except title, we picked the 10 most frequent pvs and sku: `["颜色分类", "货号", "型号", "品牌", "尺寸", "口味", "品名", "批准文号", "系列", "尺码"]`.

### About Pre-trained Models

- For image,  we use [Swin Transformer Large](https://huggingface.co/microsoft/swin-large-patch4-window12-384-in22k) pre-trained on ImageNet-22k.
- For text, we use [RoBERTa Base](https://huggingface.co/hfl/chinese-roberta-wwm-ext) pre-trained on [EXT data](https://github.com/ymcui/Chinese-BERT-wwm#:~:text=%5B1%5D%20EXT%E6%95%B0%E6%8D%AE%E5%8C%85%E6%8B%AC%EF%BC%9A%E4%B8%AD%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%EF%BC%8C%E5%85%B6%E4%BB%96%E7%99%BE%E7%A7%91%E3%80%81%E6%96%B0%E9%97%BB%E3%80%81%E9%97%AE%E7%AD%94%E7%AD%89%E6%95%B0%E6%8D%AE%EF%BC%8C%E6%80%BB%E8%AF%8D%E6%95%B0%E8%BE%BE5.4B%E3%80%82).
- Both pre-trained models are from [Hugging Face](https://huggingface.co/).

### About Model Ensemble

- We do **not** ensemble models and all results are from a single model.

### About Runtime Environment

|   GPU   | NVIDIA A100-SXM4-80GB * 2 |
| :-----: | :-----------------------: |
| Python  |           3.8.8           |
| PyTorch |           1.8.1           |
|  CUDA   |           11.1            |
|  cuDNN  |             8             |

### About Training Time and GPU Memory

|   Stage   | Training time                                                | GPU memory |
| :-------: | ------------------------------------------------------------ | :--------: |
|   Train   | Full steps, 100k iters, ~23 hours<br />Peak performance, 64k iters, ~15 hours |   ~42GB    |
| Inference | ~7 minutes                                                   |   ~16GB    |

> Train with FP16: [FP16-version](https://github.com/hanchenchen/CCKS2022-track2-solution/tree/FP16)

## 🐾 Updates

- Add emojis

## 🚧 TODO

- [x] Docker image
- [x] Pre-trained models
- [x] Logs
- [x] Results
- [x] Figure
- [x] FP16
- [x] Emoji

## 🏪 Model Zoo

|                            Model                             | Threshold |      Val<br />F1 / P / R       |     Test A<br />F1 / P / R     |                    Test B<br />F1 / P / R                    |                         Training Log                         |                             YAML                             |
| :----------------------------------------------------------: | :-------: | :----------------------------: | :----------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [63_grad_clip_norm_0.5_net_64000.pth](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/tag/1.0) |     0     | 0.8834<br />0.8909<br />0.8761 | 0.8888<br />0.8762<br />0.9017 | [0.8909<br />0.8790<br />0.9031](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/download/1.0/63_grad_clip_norm_0.5_net_64000_test_B_result_thres_0.jsonl) | [log](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/download/1.0/63_grad_clip_norm_0.5.log) | [yaml](https://github.com/hanchenchen/CCKS2022-track2-solution/blob/master/options/63_grad_clip_norm_0.5.yml) |
|                                                              |   1.65    |               -                |               -                | [0.8936<br />0.8970<br />0.8902](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/download/1.0/63_grad_clip_norm_0.5_net_64000_test_B_result_thres_1.65.jsonl) |                                                              |                                                              |
| [64_grad_clip_norm_0.1_net_60000.pth](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/tag/1.0) |     0     | 0.8753<br />0.9002<br />0.8517 | 0.8910<br />0.8901<br />0.8919 | [0.8933<br />0.8933<br />0.8933](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/download/1.0/64_grad_clip_norm_0.1_net_60000_test_B_result_thres_0.jsonl) | [log](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/download/1.0/64_grad_clip_norm_0.1.log) | [yaml](https://github.com/hanchenchen/CCKS2022-track2-solution/blob/master/options/64_grad_clip_norm_0.1.yml) |

## 🪄 Environment Setup

### Docker

- We recommend to use our established docker image [ccks-2022](registry.cn-hangzhou.aliyuncs.com/ccks-2022/ccks-2022:v1.0), which also includes our preprocessed data.

### Pip

1. Please install [PyTorch](https://pytorch.org/) according to [About Runtime Environment](#about-runtime-environment) first.
2. Then install other dependencies by `pip`.

```bash
pip install -r requirements.txt
```

## 🗺 Dataset Preparation

### Docker

- Our docker image [ccks-2022](registry.cn-hangzhou.aliyuncs.com/ccks-2022/ccks-2022:v1.0) includes our preprocessed data, which is relatively smaller and easier to download.

### Download and Preprocess manually

```bash
export REPO_DIR=$PWD

mkdir /data
cd /data
bash $REPO_DIR/scripts/download_data.sh
cat item_train_images.zip.part* > item_train_images.zip

cd $REPO_DIR
bash scripts/resize_img.sh
bash scripts/prepare_data.sh
```

## 🚄 Train

```bash
bash train.sh
```

## 📋 Test

Due to the file size limit of GitHub Release, we have to split the checkpoint. Please download [63_grad_clip_norm_0.5_net_64000.pth.partaa](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/download/1.0/63_grad_clip_norm_0.5_net_64000.pth.partaa) and [63_grad_clip_norm_0.5_net_64000.pth.partab](https://github.com/hanchenchen/CCKS2022-track2-solution/releases/download/1.0/63_grad_clip_norm_0.5_net_64000.pth.partab) to this repo and run

```bash
cat 63_grad_clip_norm_0.5_net_64000.pth.part* > 63_grad_clip_norm_0.5_net_64000.pth
bash predict.sh
```

##  📝Citations

If it helps your research or work, please consider citing our paper.
The following is a BibTeX reference. 
```
  @misc{https://doi.org/10.48550/arxiv.2208.11064,
    doi = {10.48550/ARXIV.2208.11064},
    url = {https://arxiv.org/abs/2208.11064},
    author = {Han Chenchen and Jia Heng},
    keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Multi-Modal Representation Learning with Self-Adaptive Thresholds for Commodity Verification},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
  }
```
