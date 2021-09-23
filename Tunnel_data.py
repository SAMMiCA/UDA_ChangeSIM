import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from utils import Object_Labeling
from torchvision.transforms import ToTensor, Resize, Compose
import pdb


class TunnelDataset(data.Dataset):
    def __init__(self, path, crop_size=(256, 256), ignore_label=255, num_classes=32, max_iters=None, set='train', mode='normal'):
        self.path = path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.num_classes = num_classes

        if set == 'train':
            if mode == 'target':
                self.image_total_files = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path))
                                          for f in fn if (('/semantic_segmentation' in dp) and (not '/t0' in dp))])
            else:
                self.image_total_files = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path))
                                          for f in fn if (('Seq_0' in dp) and ('/semantic_segmentation' in dp) and (not 'fire' in dp))])
        else:
            if mode == 'target':
                self.image_total_files = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path))
                                                 for f in fn if '/semantic_segmentation' in dp])
            else:
                self.image_total_files = sorted([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(path))
                                                 for f in fn if (('/semantic_segmentation' in dp) and (not 'fire' in dp))])

        if not max_iters == None:
            self.image_total_files = self.image_total_files * int(np.ceil(float(max_iters) / len(self.image_total_files)))
            self.image_total_files = self.image_total_files[:max_iters]

        self.seg = Object_Labeling.SegHelper(idx2color_path='./utils/idx2color.txt', num_class=self.num_classes)

        self.transform = Compose([Resize(crop_size), ToTensor()])

    def __len__(self):
        return len(self.image_total_files)

    def __getitem__(self, index):
        semantic_segmentation_path_mapping = self.image_total_files[index]
        image_path = semantic_segmentation_path_mapping.replace('semantic_segmentation', 'rgb')

        image = Image.open(image_path)
        label = Image.open(semantic_segmentation_path_mapping)

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        semantic_segmentation_mapping = np.asarray(label).copy()
        classmap_mapping = self.seg.colormap2classmap(semantic_segmentation_mapping)

        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]

        # image = self.transform(image)

        image = image.transpose((2,0,1)) / 255.0
        label = classmap_mapping.permute(2,0,1).squeeze(0).long()
        if (label > 31).sum() > 0:
            print(image_path)

        return image, label, semantic_segmentation_path_mapping


if __name__ == '__main__':
    dst = TunnelDataset('../../dataset/CD_Dataset', crop_size=(512, 512), ignore_label=255, num_classes=32)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        print(labels.shape)
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()