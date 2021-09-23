import torch.nn as nn
from models.network import define_UNet
import pdb

class U_net(nn.Module):
    def __init__(self, input='rit_data'):
        super(U_net, self).__init__()
        if input == 'tsunami':
            self.encoder_t, self.decoder_t = define_UNet(num_layers=7)
        elif input == 'rit_data':
            self.encoder_t, self.decoder_t = define_UNet(num_layers=8, num_classes=32)
        else:
            self.encoder_t, self.decoder_t = define_UNet()


    def forward(self, image):
        latent_t, skip_connection = self.encoder_t(image)
        output = self.decoder_t(latent_t, skip_connection)
        
        return output
