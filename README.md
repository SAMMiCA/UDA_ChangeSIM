# Unsupervised Domain Adaptation for Real-World Scenario of SAMMiCA
This repository is the official implementation of SAMMiCA scenario Unsupervised Domain Adaptation for Tunnel (ChangeSIM dataset).


## Requirements

To install requirements:

- [PyTorch](https://pytorch.org/) (An open source deep learning platform)


## Training

To train the proposed UDA method

```train
python train_CUDA.py
```

## Evaluation

To evaluate my model on CityScapes, run:

```eval
python test_CUDA.py
```


## Pre-trained Models

You can download pretrained models here:

- [pre-trained DeepLabV2 model](http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth)


## Results

Our model achieves the following performance on [ChangeSIM](https://www.cityscapes-dataset.com/) dataset:

## Acknowledgments
AI 28 Project
