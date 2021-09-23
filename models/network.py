import torch
import torch.nn as nn
from torch.nn import init
from .custom_layers import EncoderInput, DownConvolution, EncoderOutput
from .custom_layers import DecoderInput, UpConvolution, DecoderOutput
from .custom_layers import AdvancedDecoderOutput
import functools


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net



class SkipConnectionEncode(nn.Module):

    def __init__(self, norm_layer="Batch", input_channel=3, out_channel=512, num_layers=8, ngf=64):
        super(SkipConnectionEncode, self).__init__()

        if norm_layer == "Instance":
            self.use_bias = True
            self.norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_layer == "Batch":
            self.use_bias = False
            self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_layer == 'Spectral':
            self.use_bias = True
            self.norm_layer = norm_layer

        self.num_layers = num_layers
        down_list = list()
        down_list.append(EncoderInput(input_channel, ngf, self.use_bias))
        down_list.append(DownConvolution(ngf, ngf * 2, self.use_bias, self.norm_layer))
        down_list.append(DownConvolution(ngf * 2, ngf * 4, self.use_bias, self.norm_layer))
        down_list.append(DownConvolution(ngf * 4, ngf * 8, self.use_bias, self.norm_layer))
        for i in range(num_layers-5):
            down_list.append(DownConvolution(512, 512, self.use_bias, self.norm_layer))
        self.down_list = nn.ModuleList(down_list)
        self.out = EncoderOutput(512, out_channel, self.use_bias)

    def forward(self, x):
        skip_connection = list()
        out = x
        for i, module in enumerate(self.down_list):
            out = module(out)
            skip_connection.append(out)
        skip_connection.reverse()
        out = self.out(out)

        return out, skip_connection



class SkipConnectionDecode(nn.Module):

    def __init__(self, norm_layer="Batch", out_channel=3, num_layers=8,
                 use_dropout=False, transpose=True, use_advanced=False, in_channel=512):
        super(SkipConnectionDecode, self).__init__()

        if norm_layer == "Instance":
            self.use_bias = True
            self.norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_layer == "Batch":
            self.use_bias = False
            self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_layer == 'Spectral':
            self.use_bias = True
            self.norm_layer = norm_layer

        self.use_advanced = use_advanced
        self.num_layers = num_layers
        up_list = list()
        up_list.append(DecoderInput(512, 512, self.use_bias, self.norm_layer, transpose))
        for i in range(num_layers - 5):
            up_list.append(
                UpConvolution(in_channel * 2, in_channel, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(
            UpConvolution(in_channel * 2, in_channel // 2, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(
            UpConvolution(in_channel, in_channel // 4, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(
            UpConvolution(in_channel // 2, in_channel // 8, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(DecoderOutput(in_channel // 4, out_channel, transpose))
        self.up_list = nn.ModuleList(up_list)
        if self.use_advanced:
            self.advanced = AdvancedDecoderOutput(128, out_channel, self.use_bias, self.norm_layer)

    def forward(self, x, skip_connection):
        out = x.clone()
        for i, module in enumerate(self.up_list):
            if i != 0:
                out = torch.cat((skip_connection[i-1], out), 1)
            out = module(out)
        return out


def define_UNet(norm='Batch', num_layers=8, use_dropout=False,
                transpose=False, init_type='xavier', init_gain=0.02, num_classes=3):
    
    encoder = init_weights(SkipConnectionEncode(norm, 3, 512, num_layers, ngf=64),
                                  init_type, init_gain)

    decoder = init_weights(
        SkipConnectionDecode(norm, num_classes, num_layers, use_dropout, transpose, in_channel=512),
        init_type, init_gain)

    return encoder, decoder



