import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_channels=6):  
        super(Discriminator, self).__init__()

        def conv_block(in_channels, out_channels, stride=2, norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(input_channels, 64, norm=False),  
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  
        )

    def forward(self, hazy_img, clear_img):
        x = torch.cat((hazy_img, clear_img), dim=1)  
        return self.model(x)