import torch
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
import os.path as osp
import random
import torch.nn.functional as F
import copy
from ChangeSim_data import ChangeSimDataset
from models.deeplab_multi import Deeplab_multi
from utils.customized_function import load_from_checkpoint
from options_train import TrainOptions
import re
from Tunnel_data import TunnelDataset


num_workers = 4
MAX_ITERS = 250000
VAL_ITERS = 500
SAVE_PERIOD = 5000

num_classes = 32
input_size = (320, 240)
pretrained_RESNET = True
num_steps = 150000
power = 0.9
Memory_Loss = True

# data_path_train = '/media/smyoo/Backup_Data/dataset/Query_Seq_Train'
# data_path_test = '/media/smyoo/Backup_Data/dataset/Query_Seq_Test'

target_path = '../dataset/smyoo/Tunnel/Room_0/Seq_1_fire'
data_path = '../dataset/smyoo/Tunnel/Room_0/Seq_1'

saved_dir = './snapshots/SourceOnly_Tunnel'
PRE_TRAINED_SEG = './snapshots/SourceOnly/Iter_50000.pth'

SAVE_NUMPY = False
listup = False


def main():
    args = TrainOptions().parse()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.enabled = True

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    val_dataset_s = TunnelDataset(data_path, crop_size=input_size, num_classes=num_classes, ignore_label=255,
                                   set='test', mode='source')

    # DataLoader
    val_loader_s = data.DataLoader(val_dataset_s,
                                   batch_size=4, shuffle=False, num_workers=2,
                                   pin_memory=True)

    val_dataset_t = TunnelDataset(target_path, crop_size=input_size, num_classes=num_classes, ignore_label=255, set='test', mode='target')


    # DataLoader
    val_loader_t = data.DataLoader(val_dataset_t,
                                   batch_size=4, shuffle=False, num_workers=2,
                                   pin_memory=True)

    model = Deeplab_multi(args=args)

    models_path = sorted(os.listdir(saved_dir), key = lambda x: int(x.split('_')[1].split('.')[0]))

    for model_path in models_path:
        # if int(model_path.split('_')[1].split('.')[0]) < 70000:
        #     continue
        pre_trained = osp.join(saved_dir, model_path)
        print('Loaded from {}'.format(osp.join(saved_dir, model_path)))
        saved_state_dict = torch.load(pre_trained, map_location=device)
        model = load_from_checkpoint(model, saved_state_dict['state_dict_G'])
        loaded_iter = saved_state_dict['iter']
        model.to(device)
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
        model.eval()
        with torch.no_grad():
            count = 0
            total = 0
            count_s = 0
            total_s = 0

            # hist = np.zeros((num_classes, num_classes))
            # for i, test_batch in enumerate(val_loader_s):
            #     test_images, test_labels, _ = test_batch
            #     test_images = test_images.to(device)
            #     test_labels = test_labels.to(device)
            #
            #     pred2, pred1, pred_ori2, pred_ori1, _, _ = model(test_images, input_size)
            #
            #     if args.tm:
            #         pred = interp(pred2)
            #     else:
            #         pred = interp(pred_ori2)
            #
            #     _, pred = pred.max(dim=1)
            #
            #     test_labels = test_labels.cpu().numpy()
            #     pred = pred.cpu().detach().numpy()
            #
            #     hist += fast_hist(test_labels.flatten(), pred.flatten(), num_classes)
            #
            #     count_s += (pred == test_labels).sum()
            #     total_s += test_labels.shape[0] * test_labels.shape[1] * test_labels.shape[2]
            #
            # mIoUs = per_class_iu(hist)
            # mIoU = round(np.nanmean(mIoUs) * 100, 2)
            # if listup:
            #     for ind_class in range(32):
            #         print('==>' + val_dataset_t.seg.idx2name[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            # print('===> mIoU (Source): ' + str(mIoU))
            # print('===> Pixel Accuracy (Source): {}%'.format(float(count_s / total_s) * 100))

            hist = np.zeros((num_classes, num_classes))
            for i, test_batch in enumerate(val_loader_t):
                test_images, test_labels, names = test_batch
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                pred2, pred1, pred_ori2, pred_ori1, _, _ = model(test_images, input_size)
                if args.tm:
                    pred = interp(pred2)
                else:
                    pred = interp(pred_ori2)
                _, pred = pred.max(dim=1)

                if SAVE_NUMPY:
                    for i, name in enumerate(names):
                        np_path = name.replace('semantic_segmentation', 'before_UDA').replace('.png', '.npy')
                        print(np_path)
                        os.makedirs(os.path.dirname(np_path), exist_ok=True)
                        np.save(np_path, pred_ori2[i].cpu().numpy())

                test_labels = test_labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(test_labels.flatten(), pred.flatten(), num_classes)

                count += (pred == test_labels).sum()
                total += test_labels.shape[0] * test_labels.shape[1] * test_labels.shape[2]

            mIoUs = per_class_iu(hist)
            mIoU = round(np.nanmean(mIoUs) * 100, 2)

            if listup:
                for ind_class in range(32):
                    print('==>' + val_dataset_t.seg.idx2name[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Target): ' + str(mIoU))
            print('===> Pixel Accuracy (Target): {}%'.format(float(count/total) * 100))
            print('\n')


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main()
