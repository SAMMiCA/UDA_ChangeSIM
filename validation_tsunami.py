import torch
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import random
import copy
from options_AI28 import TrainOptions
from dataset.cityscapes_dataset import CityScapesDataSet
from models.U_net import U_net
from torchvision import transforms
from dataset.change_dataset import ChangeDatasetNumpy



PRE_TRAINED_SEG = ''

args = TrainOptions().parse()

model_path = './snapshots/SkipAE/80.pth'


def main():
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    img_size = (128, 512)
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            # helper_augmentations.SwapReferenceTest(),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            # helper_augmentations.JitterGamma(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    # DataLoader
    val_path = '/home/smyoo/Downloads/TSUNAMI/train'
    val_dataset = ChangeDatasetNumpy(val_path, data_transforms['train'])

    # DataLoader
    targetloader = data.DataLoader(val_dataset,
                                   batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)


    # Create network
    model = U_net(tsunami=True)
    
    saved_state_dict = torch.load(model_path)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        if i in new_params.keys():
            new_params[i] = saved_state_dict[i]
    model.load_state_dict(new_params)
    
    model.to(device)

    for num, batch in enumerate(targetloader, 1):
        # train f
        samples = batch
        images = samples['reference']
        noise = np.random.choice(2, size=(images.shape[0], images.shape[1], images.shape[2], images.shape[3]),
                                 p=[0.6, 0.4])

        im_ori = transforms.ToPILImage()(images.squeeze(0).cpu()).convert("RGB")
        im_ori.save('plots/%s_input.png' % (str(num)))

        images = images * noise

        images = images.to(device).float()
        output = model(images)

        im_input = transforms.ToPILImage()(images.squeeze(0).cpu()).convert("RGB")
        im_input.save('plots/%s_input_noise.png' % (str(num)))

        im = transforms.ToPILImage()(output.squeeze(0).cpu()).convert("RGB")
        im.save('plots/%s_reconstruct.png' % (str(num)))
  

if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
