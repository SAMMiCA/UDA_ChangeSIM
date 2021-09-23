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
from models.deeplabv2.deeplab_multi import DeeplabMultiFeature, DeeplabMulti
from models.deeplabv3.deeplabv3 import DeepLabV3
from models.deeplab_multi import Deeplab_multi
from models.discriminator import FCDiscriminator, DHA
from utils.customized_function import save_model, load_from_checkpoint
from options_train import TrainOptions

import pdb
from tensorboardX import SummaryWriter


num_workers = 4

num_classes = 32
input_size = (320, 240)

# data_path_train = '/media/smyoo/Backup_Data/dataset/Query_Seq_Train'
# data_path_test = '/media/smyoo/Backup_Data/dataset/Query_Seq_Test'

data_path_train = '../dataset/smyoo/Query_Seq_Train'
data_path_test = '../dataset/smyoo/Query_Seq_Test'

restore_from = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
source_mode = 'normal'
# mode = 'dust'


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_D(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr


def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if args.from_scratch:
        optimizer.param_groups[1]['lr'] = lr
    else:
        optimizer.param_groups[1]['lr'] = lr * 10
    if args.tm:
        optimizer.param_groups[2]['lr'] = lr * 10


def distillation_loss(pred_origin, old_outputs):
    pred_origin_logsoftmax = (pred_origin / 2).log_softmax(dim=1)
    old_outputs = (old_outputs / 2).softmax(dim=1)
    loss_distillation = (-(old_outputs * pred_origin_logsoftmax)).sum(dim=1)
    loss_distillation = loss_distillation.sum() / loss_distillation.flatten().shape[0]
    return loss_distillation


def prob_2_entropy(prob):
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)



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
    log_path = osp.join('./logs', args.dir_name)
    writer = SummaryWriter(log_path)
    source_dataset = ChangeSimDataset(data_path_train, crop_size=input_size, num_classes=num_classes, ignore_label=255, max_iters=args.num_steps * args.batch_size, mode=source_mode)
    target_dataset = ChangeSimDataset(data_path_train, crop_size=input_size, num_classes=num_classes, ignore_label=255, max_iters=args.num_steps * args.batch_size, mode=args.target)

    val_dataset = ChangeSimDataset(data_path_test, crop_size=input_size, ignore_label=255, max_iters=200, num_classes=num_classes, mode=args.target)


    # DataLoader
    sourceloader = data.DataLoader(source_dataset,
                                   batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                   pin_memory=True, drop_last=True)
    targetloader = data.DataLoader(target_dataset,
                                    batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                                    pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset,
                                   batch_size=4, shuffle=False, num_workers=2,
                                   pin_memory=True)

    source_loader_iter = enumerate(sourceloader)
    target_loader_iter = enumerate(targetloader)

    model = Deeplab_multi(args=args)

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)
    model_D2 = FCDiscriminator(num_classes=args.num_classes).to(device)

    model_D1.train()
    model_D1.to(device)

    model_D2.train()
    model_D2.to(device)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

    if args.from_scratch:  # training model from pre-trained ResNet
        saved_state_dict = torch.load(args.restore_from_resnet, map_location=device)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    else:  # training model from pre-trained DeepLabV2 on source & previous target domains
        # saved_state_dict = torch.load(args.pre_trained_seg, map_location=device)
        # new_params = model.state_dict().copy()
        # for i in saved_state_dict:
        #     if i in new_params.keys():
        #         new_params[i] = saved_state_dict[i]
        # model.load_state_dict(new_params)
        saved_state_dict = torch.load(args.pre_trained_seg, map_location=device)

        model = load_from_checkpoint(model, saved_state_dict['state_dict_G'])
        # optimizer = load_from_checkpoint(optimizer, saved_state_dict['optimizer_G'])
        # loaded_iter = saved_state_dict['iter']
        # model_D1 = load_from_checkpoint(model, saved_state_dict['state_dict_D1'])
        # optimizer_D1 = load_from_checkpoint(optimizer, saved_state_dict['optimizer_D1'])
        # model_D2 = load_from_checkpoint(model, saved_state_dict['state_dict_D2'])
        # optimizer_D2 = load_from_checkpoint(optimizer, saved_state_dict['optimizer_D2'])

    model.train()
    model.to(device)

    # reference model
    if not args.from_scratch:
        ref_model = copy.deepcopy(model)  # reference model for knowledge distillation
        for params in ref_model.parameters():
            params.requires_grad = False
        ref_model.eval()

    optimizer.zero_grad()
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'DHA':
        adversarial_loss_1 = DHA(model_D1, relative=args.relative)
        adversarial_loss_2 = DHA(model_D2, relative=args.relative)
    else:
        raise NotImplementedError('Unavailable GAN option')

    # labels for adversarial training
    source_label = 1
    target_label = 0

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # Snapshots directory
    if not os.path.exists(osp.join(args.snapshot_dir, args.dir_name)):
        os.makedirs(osp.join(args.snapshot_dir, args.dir_name))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # start training
    for i_iter in range(1, args.num_steps_stop):

        loss_seg_value1 = 0
        loss_adv_value1 = 0
        loss_distill_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_value2 = 0
        loss_distill_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter, args)
        adjust_learning_rate_D(optimizer_D2, i_iter, args)

        # train f

        # freeze D
        for param in model_D1.parameters():
            param.requires_grad = False

        for param in model_D2.parameters():
            param.requires_grad = False

        _, batch = source_loader_iter.__next__()

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        if args.tm:
            pred2, pred1, pred_ori2, pred_ori1, pred2_tm, pred1_tm = model(images, input_size)
        else:
            _, _, pred2, pred1, _, _ = model(images, input_size)

        loss_seg1 = seg_loss(pred1, labels)
        loss_seg2 = seg_loss(pred2, labels)
        if args.memory_loss:
            loss_seg1_mem = seg_loss(pred1_tm, labels)
            loss_seg2_mem = seg_loss(pred2_tm, labels)
            loss = args.lambda_seg1 * loss_seg1 + args.lambda_seg2 * loss_seg2 + args.lambda_seg1 * 0.5 * loss_seg1_mem + args.lambda_seg2 * 0.5 * loss_seg2_mem
        else:
            loss = args.lambda_seg1 * loss_seg1 + args.lambda_seg2 * loss_seg2

        loss_seg_value1 += loss_seg1.item()
        loss_seg_value2 += loss_seg2.item()

        # if not args.from_scratch and args.tm:
        #     _, _, old_outputs2, old_outputs1 = ref_model(images, input_size)
        #     loss_distill1 = distillation_loss(pred_ori1, old_outputs1)
        #     loss_distill2 = distillation_loss(pred_ori2, old_outputs2)
        #     loss += args.lambda_distill1 * loss_distill1 + args.lambda_distill2 * loss_distill2
        #     loss_distill_value1 += loss_distill1.item()
        #     loss_distill_value2 += loss_distill2.item()

        if not args.gan == 'DHA':
            loss.backward()

        _, batch = target_loader_iter.__next__()
        images_target, _, _ = batch
        images_target = images_target.to(device)

        if args.tm:
            pred2_target, pred1_target, _, _, pred2_target_tm, pred1_target_tm = model(images_target, input_size)
        else:
            _, _, pred2_target, pred1_target, _, _ = model(images_target, input_size)

        if args.gan == 'DHA':
            if args.ent:
                loss_adv1 = adversarial_loss_1(prob_2_entropy(F.softmax(pred1_target, dim=1)),
                                               prob_2_entropy(F.softmax(pred1, dim=1)),
                                               loss_type='adversarial')
                loss_adv2 = adversarial_loss_2(prob_2_entropy(F.softmax(pred2_target, dim=1)),
                                               prob_2_entropy(F.softmax(pred2, dim=1)),
                                               loss_type='adversarial')
            else:
                loss_adv1 = adversarial_loss_1(F.softmax(pred1_target, dim=1),
                                               F.softmax(pred1, dim=1),
                                               loss_type='adversarial')
                loss_adv2 = adversarial_loss_2(F.softmax(pred2_target, dim=1),
                                               F.softmax(pred2, dim=1),
                                               loss_type='adversarial')
            if args.memory_loss:
                loss_adv1_mem = adversarial_loss_1(F.softmax(pred1_target_tm, dim=1),
                                               F.softmax(pred1_tm, dim=1),
                                               loss_type='adversarial')
                loss_adv2_mem = adversarial_loss_2(F.softmax(pred2_target_tm, dim=1),
                                               F.softmax(pred2_tm, dim=1),
                                               loss_type='adversarial')

                loss += args.lambda_adv1 * loss_adv1 + args.lambda_adv2 * loss_adv2 + args.lambda_adv1 * 0.5 * loss_adv1_mem\
                        + args.lambda_adv2 * 0.5 * loss_adv2_mem
            else:
                loss += args.lambda_adv1 * loss_adv1 + args.lambda_adv2 * loss_adv2

        elif args.gan == 'Vanilla':
            if args.ent:
                D_out1 = model_D1(prob_2_entropy(F.softmax(pred1_target, dim=1)))
                D_out2 = model_D2(prob_2_entropy(F.softmax(pred2_target, dim=1)))
            else:
                D_out1 = model_D1(F.softmax(pred1_target, dim=1))
                D_out2 = model_D2(F.softmax(pred2_target, dim=1))

            loss_adv1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
            loss_adv2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv1 * loss_adv1 + args.lambda_adv2 * loss_adv2
        else:
            raise NotImplementedError('Unavailable GAN option')

        loss_adv_value1 += loss_adv1.item()
        loss_adv_value2 += loss_adv2.item()
        loss.backward()

        # train D

        # bring back requires_grad
        for param in model_D1.parameters():
            param.requires_grad = True

        for param in model_D2.parameters():
            param.requires_grad = True

        pred1 = pred1.detach()
        pred2 = pred2.detach()
        pred1_target = pred1_target.detach()
        pred2_target = pred2_target.detach()

        if args.tm:
            pred1_tm = pred1_tm.detach()
            pred2_tm = pred2_tm.detach()
            pred1_target_tm = pred1_target_tm.detach()
            pred2_target_tm = pred2_target_tm.detach()

        if args.gan == 'DHA':
            if args.ent:
                loss_D1 = adversarial_loss_1(prob_2_entropy(F.softmax(pred1_target, dim=1)),
                                             prob_2_entropy(F.softmax(pred1, dim=1)),
                                             loss_type='discriminator')
                loss_D2 = adversarial_loss_2(prob_2_entropy(F.softmax(pred2_target, dim=1)),
                                             prob_2_entropy(F.softmax(pred2, dim=1)),
                                             loss_type='discriminator')
            else:
                loss_D1 = adversarial_loss_1(F.softmax(pred1_target, dim=1),
                                             F.softmax(pred1, dim=1),
                                             loss_type='discriminator')
                loss_D2 = adversarial_loss_2(F.softmax(pred2_target, dim=1),
                                             F.softmax(pred2, dim=1),
                                             loss_type='discriminator')

            if args.memory_loss:
                loss_D1 += 0.5 * adversarial_loss_1(F.softmax(pred1_target_tm, dim=1),
                                             F.softmax(pred1_tm, dim=1),
                                             loss_type='discriminator')
                loss_D2 += 0.5 * adversarial_loss_2(F.softmax(pred2_target_tm, dim=1),
                                             F.softmax(pred2_tm, dim=1),
                                             loss_type='discriminator')

                # loss_D1 = loss_D1 / 1.5
                # loss_D2 = loss_D2 / 1.5

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
        elif args.gan == 'Vanilla':
            # train with source
            if args.ent:
                D_out1 = model_D1(prob_2_entropy(F.softmax(pred1, dim=1)))
                D_out2 = model_D2(prob_2_entropy(F.softmax(pred2, dim=1)))
            else:
                D_out1 = model_D1(F.softmax(pred1, dim=1))
                D_out2 = model_D2(F.softmax(pred2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()

            # train with target
            if args.ent:
                D_out1 = model_D1(prob_2_entropy(F.softmax(pred1_target, dim=1)))
                D_out2 = model_D2(prob_2_entropy(F.softmax(pred2_target, dim=1)))
            else:
                D_out1 = model_D1(F.softmax(pred1_target, dim=1))
                D_out2 = model_D2(F.softmax(pred2_target, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))

            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            loss_D_value2 += loss_D2.item()
        else:
            raise NotImplementedError('Unavailable GAN option')

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))
        print('iter = {0:8d}/{1:8d}'.format(i_iter, args.num_steps))
        print('loss_seg1 = {0:.3f} loss_dist1 = {1:.3f} loss_adv1 = {2:.3f} loss_D1 = {3:.3f}'.format(
            loss_seg_value1, loss_distill_value1, loss_adv_value1, loss_D_value1))
        print('loss_seg2 = {0:.3f} loss_dist2 = {1:.3f} loss_adv2 = {2:.3f} loss_D2 = {3:.3f}'.format(
            loss_seg_value2, loss_distill_value2, loss_adv_value2, loss_D_value2))

        if i_iter % 100 == 0:
            writer.add_scalars('Train/Seg_loss_1', {'train': loss_seg_value1}, i_iter)
            writer.add_scalars('Train/Seg_loss_2', {'train': loss_seg_value2}, i_iter)
            writer.add_scalars('Train/Adv_loss_1', {'train': loss_adv_value1}, i_iter)
            writer.add_scalars('Train/Adv_loss_2', {'train': loss_adv_value2}, i_iter)
            writer.add_scalars('Train/D_loss_1', {'train': loss_D_value1}, i_iter)
            writer.add_scalars('Train/D_loss_2', {'train': loss_D_value2}, i_iter)

        # if i_iter % 5000 == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         hist = np.zeros((num_classes, num_classes))
        #         count = 0
        #         total = 0
        #         for i, test_batch in enumerate(val_loader):
        #             test_images, test_labels, _ = test_batch
        #             test_images = test_images.to(device)
        #             test_labels = test_labels.to(device)
        #
        #             pred2, pred1, pred_ori2, pred_ori1, _, _ = model(test_images, input_size)
        #
        #             pred = interp(pred2)
        #             _, pred = pred.max(dim=1)
        #
        #             test_labels = test_labels.cpu().numpy()
        #             pred = pred.cpu().detach().numpy()
        #
        #             hist += fast_hist(test_labels.flatten(), pred.flatten(), num_classes)
        #
        #             count += (pred == test_labels).sum()
        #             total += test_labels.shape[0] * test_labels.shape[1] * test_labels.shape[2]
        #
        #         mIoUs = per_class_iu(hist)
        #         mIoU = round(np.nanmean(mIoUs) * 100, 2)
        #         for ind_class in range(32):
        #             print('==>' + source_dataset.seg.idx2name[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        #         print('===> mIoU (Test): ' + str(mIoU))
        #
        #         print('===> Pixel Accuracy (Test): {}%'.format(float(count/total) * 100))
        #
        #         model.train()

        if i_iter % args.save_pred_every == 0:
            print('save model ...')
            save_model(osp.join(args.snapshot_dir, args.dir_name), i_iter, model, optimizer, loss_seg_value2, args,
                       model_D1, optimizer_D1, model_D2, optimizer_D2)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
