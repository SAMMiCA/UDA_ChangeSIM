import os
import torch


def save_model(model_path, iter, model_G, optimizer_G, loss, opt, model_D1=None, optim_D1=None, model_D2=None, optim_D2=None):
    info = {'state_dict_G': model_G.state_dict(), 'optimizer_G': optimizer_G.state_dict(), 'iter': iter, 'loss': loss, 'opt': opt}

    # if model_D1 is not None:
    #     info['state_dict_D1'] = model_D1.state_dict()
    #     info['optimizer_D1'] = optim_D1.state_dict()
    # if model_D2 is not None:
    #     info['state_dict_D2'] = model_D2.state_dict()
    #     info['optimizer_D2'] = optim_D2.state_dict()

    model_out_path = "Iter_{}.pth".format(iter)

    if not (os.path.exists(model_path)):
        os.makedirs(model_path)
    model_out_path = os.path.join(model_path, model_out_path)
    torch.save(info, model_out_path)
    # torch.save(info, model_out_path, _use_new_zipfile_serialization=False)

    print("Checkpoint saved to {}".format(model_out_path))


def load_from_checkpoint(model, checkpoint):
    new_params = model.state_dict().copy()
    for i in checkpoint:
        if i in new_params.keys():
            new_params[i] = checkpoint[i]
    model.load_state_dict(new_params)
    return model