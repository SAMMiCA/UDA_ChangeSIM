import argparse
import numpy as np
import random

import torch
from torch.utils import data
from ChangeSim_data import ChangeSimDataset
from models.deeplabv2.deeplab_multi import DeeplabMulti
from models.deeplabv3.deeplabv3 import DeepLabV3
import os
from PIL import Image
import torch.nn as nn
from utils import Object_Labeling
import os.path as osp
import pickle
import math


BATCH_SIZE = 1
snapshot_dir = 'snapshots'
IGNORE_LABEL = 255
NUM_CLASSES = 32
SET = 'val'
RANDOM_SEED = 1338
SAVE = './plots'
deeplabv2 = True

# data_path = '/media/smyoo/Backup_Data/dataset/Indoor/Mapping'
# data_path = '/home/ai31/datasets/smyoo/Indoor/Mapping'
data_path = '/home/smyoo/Downloads/warehouse'


with open('./utils/aa.pkl', 'rb') as f:
    object_info = pickle.load(f)


available = object_info[0] + ['background']
weight = object_info[2] + [100 - sum(object_info[2])]

weight_dict = dict(zip(available, weight))

# available = ['wall', 'floor', 'box', 'shelf', 'pallet', 'background', 'column', 'pipe', 'beam', 'frame', 'fence', 'wire',
#              'ceiling', 'duct', 'lamp', 'door', 'barrel', 'sign', 'bag', 'electric_box', 'vehicle', 'ladder', 'canister',
#              'extinguisher', 'hand_truck']

if deeplabv2:
    log_path = './logs/DeepLab_V2'
    learning_rate = 1e-4
    DIR_NAME = 'DeepLab_V2'
else:
    log_path = './logs/DeepLab_V3'
    learning_rate = 1e-3
    DIR_NAME = 'DeepLab_V3'


# checkpoint = osp.join(snapshot_dir, DIR_NAME, 'deeplab_v3_nozip.pth')
checkpoint = osp.join(snapshot_dir, DIR_NAME, '20.pth')



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Indoor Dataset Plot")

    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                             help="Number of images sent to the network in one step.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--dir-name", type=str, default=DIR_NAME)
    parser.add_argument("--save", type=str, default=SAVE)

    return parser.parse_args()

# palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
#            220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
#            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

palette = []

Seg_Helper = Object_Labeling.SegHelper(idx2color_path='./utils/idx2color.txt', num_class=NUM_CLASSES)

color2idx = Seg_Helper.color2idx
for color, cls in color2idx.items():
    palette += list(color)



# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = (640, 480)

    # Create network
    if deeplabv2:
        model = DeeplabMulti(num_classes=NUM_CLASSES)
    else:
        model = DeepLabV3(num_classes=NUM_CLASSES)

    checkpoint_dict = torch.load(checkpoint)
    model.load_state_dict(checkpoint_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)


    if not os.path.exists(os.path.join(args.save, args.dir_name.split('/')[0], 'Indoor')):
        os.makedirs(os.path.join(args.save, args.dir_name.split('/')[0], 'Indoor'))

    test_dataset = ChangeSimDataset(data_path, crop_size=input_size, ignore_label=255,
                                    num_classes=NUM_CLASSES, set='test')

    test_dataset = ChangeSimDataset(data_path, crop_size=input_size, ignore_label=255,
                                    num_classes=NUM_CLASSES, set='custom')


    Indoor_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    with torch.no_grad():
        hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
        count = 0
        total = 0
        for index, batch in enumerate(Indoor_loader):
            if index % 100 == 0:
                print('%d processed (Indoor)' % index)
            image, test_labels, name = batch
            image = image.to(device)

            if deeplabv2:
                _, pred = model(image)
            else:
                _, _, _, pred = model(image)
            pred = interp(pred)

            _, top1 = pred.max(dim=1)
            test_labels = test_labels.cpu().numpy()
            top1 = top1.cpu().detach().numpy()

            hist += fast_hist(test_labels.flatten(), top1.flatten(), NUM_CLASSES)

            count += (top1 == test_labels).sum()
            total += test_labels.shape[0] * test_labels.shape[1] * test_labels.shape[2]
            print('===> Pixel Accuracy (Test): {}%'.format(float(count / total) * 100))

            output = pred.cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)

            name = name[0].split('/')[-1]
            output_col.save('%s/%s/Indoor/%s_color.png' % (args.save, args.dir_name.split('/')[0],
                                                               name.split('.')[0]))

        mIoUs = per_class_iu(hist)
        mIoU = round(np.nanmean(mIoUs) * 100, 2)

        fwIoU_list = []
        mIoU_list = []

        for ind_class in range(32):
            class_name = test_dataset.seg.idx2name[ind_class]
            if class_name in available:
                class_IoU = round(mIoUs[ind_class] * 100, 2)
                if math.isnan(class_IoU):
                    class_IoU = 0.0
                fwIoU_list.append(weight_dict[class_name] * class_IoU)
                mIoU_list.append(class_IoU)

                print('==>' + test_dataset.seg.idx2name[ind_class] + ':\t' + str(class_IoU))

        print('===> mIoU (Test): ' + str(sum(mIoU_list) / 25))

        print('===> fwIoU (Test): ' + str(sum(fwIoU_list) / 100))

        print('===> Total Pixel Accuracy (Test): {}%'.format(float(count / total) * 100))



if __name__ == '__main__':
    main()