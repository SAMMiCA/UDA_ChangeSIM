import torch.nn as nn


class SpectralConvolution(nn.Module):

    def __init__(self, in_ch, out_ch, w, s, p, bias):
        super(SpectralConvolution, self).__init__()
        self.l = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=w, stride=s, padding=p, bias=bias))

    def forward(self, x):
        return self.l(x)


class SpectralTranspose(nn.Module):

    def __init__(self, in_ch, out_ch, w, s, p, bias):
        super(SpectralTranspose, self).__init__()
        self.l = nn.utils.spectral_norm(nn.ConvTranspose2d(in_ch, out_ch,
                                                           kernel_size=w, stride=s, padding=p, bias=bias))

    def forward(self, x):
        return self.l(x)


class NaiveConvolution(nn.Module):

    def __init__(self, in_ch, out_ch, w, s, p, bias, norm_layer):
        super(NaiveConvolution, self).__init__()
        self.l = nn.Conv2d(in_ch, out_ch, kernel_size=w, stride=s, padding=p, bias=bias)
        self.norm = norm_layer(out_ch)

    def forward(self, x):
        return self.norm(self.l(x))


class NaiveTranspose(nn.Module):

    def __init__(self, in_ch, out_ch, w, s, p, bias, norm_layer):
        super(NaiveTranspose, self).__init__()
        self.l = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=w, stride=s, padding=p, bias=bias)
        self.norm = norm_layer(out_ch)

    def forward(self, x):
        return self.norm(self.l(x))


"""
For Encoder
"""
class EncoderInput(nn.Module):

    def __init__(self, in_ch, out_ch, use_bias):
        super(EncoderInput, self).__init__()
        self.input = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=use_bias)

    def forward(self, x):
        out = self.input(x)
        return out


class DownConvolution(nn.Module):

    def __init__(self, in_ch, out_ch, use_bias, norm_layer):
        super(DownConvolution, self).__init__()
        sequence = [nn.LeakyReLU(0.2, True)]
        if norm_layer == 'Spectral':
            sequence += [SpectralConvolution(in_ch, out_ch, w=4, s=2, p=1, bias=use_bias)]
        else:
            sequence += [NaiveConvolution(in_ch, out_ch, w=4, s=2, p=1, bias=use_bias, norm_layer=norm_layer)]
        self.block = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.block(x)
        return out


class EncoderOutput(nn.Module):

    def __init__(self, in_ch, out_ch, use_bias):
        super(EncoderOutput, self).__init__()
        self.down = nn.Sequential(nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(in_ch, out_ch, kernel_size=4,
                                            stride=2, padding=1, bias=use_bias))

    def forward(self, x):
        out = self.down(x)
        return out


"""
For Decoder
"""

class DecoderInput(nn.Module):

    def __init__(self, in_ch, out_ch, use_bias, norm_layer, use_transpose):
        super(DecoderInput, self).__init__()

        if use_transpose:
            sequence = [nn.ReLU(True)]
            if norm_layer == 'Spectral':
                sequence += [SpectralTranspose(in_ch, out_ch, w=4, s=2, p=1, bias=use_bias)]
            else:
                sequence += [NaiveTranspose(in_ch, out_ch, w=4, s=2, p=1, bias=use_bias, norm_layer=norm_layer)]
        else:
            sequence = [nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
            if norm_layer == 'Spectral':
                sequence += [SpectralConvolution(in_ch, out_ch, w=3, s=1, p=1, bias=use_bias)]
            else:
                sequence += [NaiveConvolution(in_ch, out_ch, w=3, s=1, p=1, bias=use_bias, norm_layer=norm_layer)]

        self.block = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.block(x)
        return out


class UpConvolution(nn.Module):

    def __init__(self, in_ch, out_ch, use_bias, norm_layer, use_dropout, use_transpose):
        super(UpConvolution, self).__init__()

        if use_transpose:
            sequence = [nn.ReLU(True)]
            if norm_layer == 'Spectral':
                sequence += [SpectralTranspose(in_ch, out_ch, w=4, s=2, p=1, bias=use_bias)]
            else:
                sequence += [NaiveTranspose(in_ch, out_ch, w=4, s=2, p=1, bias=use_bias, norm_layer=norm_layer)]
        else:
            sequence = [nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
            if norm_layer == 'Spectral':
                sequence += [SpectralConvolution(in_ch, out_ch, w=3, s=1, p=1, bias=use_bias)]
            else:
                sequence += [NaiveConvolution(in_ch, out_ch, w=3, s=1, p=1, bias=use_bias, norm_layer=norm_layer)]

        self.block = nn.Sequential(*sequence)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.block(x)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class DecoderOutput(nn.Module):

    def __init__(self, in_ch, out_ch, use_transpose):
        super(DecoderOutput, self).__init__()

        if use_transpose:
            self.output = nn.Sequential(nn.ReLU(True),
                                        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                                                           stride=2, padding=1, bias=True))
        else:
            self.output = nn.Sequential(nn.ReLU(True),
                                        nn.Upsample(scale_factor=2, mode='bilinear'),
                                        nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                                  stride=1, padding=1, bias=True))

    def forward(self, x):
        out = self.output(x)
        return out


class AdvancedDecoderOutput(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias, norm_layer):
        super(AdvancedDecoderOutput, self).__init__()

        # Change channel to 64
        input_sequence = [nn.ReLU(True)]
        if norm_layer == 'Spectral':
            input_sequence += [SpectralConvolution(in_ch, 64, w=3, s=1, p=1, bias=use_bias)]
        else:
            input_sequence += [NaiveConvolution(in_ch, 64, w=3, s=1, p=1, bias=use_bias, norm_layer=norm_layer)]
        input_sequence += [nn.PReLU()]
        self.input = nn.Sequential(*input_sequence)

        # Residual block
        if norm_layer == 'Spectral':
            residual_sequence = [SpectralConvolution(64, 64, w=3, s=1, p=1, bias=use_bias)]
        else:
            residual_sequence = [NaiveConvolution(64, 64, w=3, s=1, p=1, bias=use_bias, norm_layer=norm_layer)]
        residual_sequence += [nn.PReLU()]
        if norm_layer == 'Spectral':
            residual_sequence += [SpectralConvolution(64, 64, w=3, s=1, p=1, bias=use_bias)]
        else:
            residual_sequence += [NaiveConvolution(64, 64, w=3, s=1, p=1, bias=use_bias, norm_layer=norm_layer)]
        self.residual = nn.Sequential(*residual_sequence)

        # Pixel shuffle block
        self.output = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3,
                                              stride=1, padding=1, bias=use_bias),
                                    nn.PixelShuffle(2),
                                    nn.PReLU(),
                                    nn.Conv2d(64, out_ch, kernel_size=9,
                                              stride=1, padding=4, bias=True))

    def forward(self, x):
        out = self.input(x)
        identity = out
        res = self.residual(out)
        out = identity + res
        out = self.output(out)
        return out

