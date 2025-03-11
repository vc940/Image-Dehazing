import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader




class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), 
                                   nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), 
                                   nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), 
                                   nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), 
                                   nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU())
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1), nn.ReLU(),
                                        nn.Conv2d(1024, 1024, 3, 1, 1), nn.ReLU())

        self.upConv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv5 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU())

        self.upConv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv6 = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU())

        self.upConv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv7 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())

        self.upConv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv8 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
                                   nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU())

        self.output = nn.Conv2d(64, 3, 1)  
    def forward(self, x):
        x1 = self.conv1(x) 
        x2 = self.conv2(self.pool1(x1)) 
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3)) 
        x5 = self.bottleneck(self.pool4(x4))  
        x6 = self.upConv1(x5)
        x6 = torch.cat([x4, x6], dim=1)  
        x6 = self.conv5(x6)
        x7 = self.upConv2(x6)  
        x7 = torch.cat([x3, x7], dim=1)
        x7 = self.conv6(x7)

        x8 = self.upConv3(x7)  
        x8 = torch.cat([x2, x8], dim=1)
        x8 = self.conv7(x8)

        x9 = self.upConv4(x8)  
        x9 = torch.cat([x1, x9], dim=1)
        x9 = self.conv8(x9)

        output = self.output(x9) 
        output = torch.sigmoid(output)  

        return output
