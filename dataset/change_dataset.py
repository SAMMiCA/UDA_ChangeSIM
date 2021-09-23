from torch.utils.data import Dataset
import numpy as np
import pickle
import torchvision
import helper_augmentations
import os
from PIL import Image


class ChangeDatasetNumpy(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""

    def __init__(self, data_path, transform=None):

        # Load pickle file with Numpy dictionary

        self.data_list = sorted(os.listdir(os.path.join(data_path, 't0')))
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        reference_path = os.path.join(self.data_path, 't0', file_name)
        test_path = os.path.join(self.data_path, 't1', file_name)
        label_path = os.path.join(self.data_path, 'mask', file_name)

        reference_PIL = Image.open(reference_path)
        test_PIL = Image.open(test_path)
        label_PIL = Image.open(label_path.replace('.jpg', '.png'))

        sample = {'reference': reference_PIL, 'test': test_PIL, 'label': label_PIL}

        # Handle Augmentations
        if self.transform:
            trf_reference = sample['reference']
            trf_test = sample['test']
            trf_label = sample['label']
            # Dont do Normalize on label, all the other transformations apply...
            for t in self.transform.transforms:
                if (isinstance(t, helper_augmentations.SwapReferenceTest)) or (
                isinstance(t, helper_augmentations.JitterGamma)):
                    trf_reference, trf_test = t(sample)
                else:
                    # All other type of augmentations
                    trf_reference = t(trf_reference)
                    trf_test = t(trf_test)

                # Don't Normalize or Swap
                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    # ToTensor divide every result by 255
                    # https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#to_tensor
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        trf_label = t(trf_label) * 255.0
                    else:
                        if not isinstance(t, helper_augmentations.SwapReferenceTest):
                            if not isinstance(t, torchvision.transforms.transforms.ColorJitter):
                                if not isinstance(t, torchvision.transforms.transforms.RandomGrayscale):
                                    if not isinstance(t, helper_augmentations.JitterGamma):
                                        trf_label = t(trf_label)

            sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label}

        return sample