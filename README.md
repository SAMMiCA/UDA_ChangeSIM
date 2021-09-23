# Change Detection in 3D Reconstruction Map and the current view
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

- [Our pre-trained model](https://drive.google.com/drive/u/0/folders/1Sd2h2xbpVUH_70EGil2-Y2VYDp7HB1Es) trained on CityScapes with learning rate 1e-3, batch size 8, input size (512, 256). 


## Results

Our model achieves the following performance on [ChangeSIM](https://www.cityscapes-dataset.com/) dataset:

## Acknowledgments
AI 28 Project
