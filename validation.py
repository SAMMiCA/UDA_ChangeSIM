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


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PRE_TRAINED_SEG = ''

args = TrainOptions().parse()

model_path = './snapshots/SkipAE/9.pth'


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

    # DataLoader
    targetloader = data.DataLoader(CityScapesDataSet(args.data_dir_target, args.data_list_target_val, crop_size=input_size, set='val'),
                                   batch_size=1, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)

    # Create network
    model = U_net()
    
    saved_state_dict = torch.load(model_path)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        if i in new_params.keys():
            new_params[i] = saved_state_dict[i]
    model.load_state_dict(new_params)
    
    model.to(device)

    for num, batch in enumerate(targetloader, 1):
        # train f
        images, labels = batch
        images = images.to(device)
        output = model(images)

        im_input = transforms.ToPILImage()(images.squeeze(0).cpu()).convert("RGB")
        im_input.save('plots/%s_input.png' % (str(num)))

        im = transforms.ToPILImage()(output.squeeze(0).cpu()).convert("RGB")
        im.save('plots/%s_reconstruct.png' % (str(num)))
  

if __name__ == '__main__':
    main()
