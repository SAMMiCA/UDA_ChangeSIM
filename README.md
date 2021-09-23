# Change Detection in 3D Reconstruction Map and the current view
This repository is the official implementation of SAMMiCA scenario 1 3D Change Detection.


## Requirements

To install requirements:

- [PyTorch](https://pytorch.org/) (An open source deep learning platform)


## Training

To train the U-net architecture model for the inpainting task, run this command:

```train
python train.py
```

## Evaluation

To evaluate my model on CityScapes, run:

```eval
python validation.py
```


## Pre-trained Models

You can download pretrained models here:

- [Our pre-trained model](https://drive.google.com/drive/u/0/folders/1Sd2h2xbpVUH_70EGil2-Y2VYDp7HB1Es) trained on CityScapes with learning rate 1e-3, batch size 8, input size (512, 256). 


## Results

Our model achieves the following performance on [CityScapes](https://www.cityscapes-dataset.com/) dataset:

#### Input 1
![example9](examples/9_input.png)
#### Output 1
![recon9](examples/9_reconstruct.png)

#### Input 2
![example9](examples/23_input.png)
#### Output 2
![recon9](examples/23_reconstruct.png)

## Acknowledgments
AI 28 Project
