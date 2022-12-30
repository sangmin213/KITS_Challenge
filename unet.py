import torch.nn as nn
import torch
import numpy as np
import cv2

# encoding block
def CBR2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
    layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )
    return layers


# decoding block
def CBR(in_channel, mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
    layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channel,out_channels=mid_channel,
                kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        nn.Conv2d(in_channels=mid_channel,out_channels=out_channel,
                kernel_size=kernel_size,stride=stride,padding=padding,bias=bias),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )
    return layers

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.n_classes = 1     
        
        # Contracting Path
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.enc1 = CBR2d(1, 64)
        self.enc2 = CBR2d(64, 128)
        self.enc3 = CBR2d(128, 256)
        self.enc4 = CBR2d(256, 512)
        self.enc5 = nn.Sequential(
                nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(1024),
                nn.ReLU())
        
        # Expanding Path
        self.dec5 = nn.Sequential(
                nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(512),
                nn.ReLU())
        
        self.unpool4 = nn.ConvTranspose2d(512,512,kernel_size=2,stride=2,padding=0,bias=True)
        self.dec4 = CBR(1024,512,256)

        self.unpool3 = nn.ConvTranspose2d(256,256,kernel_size=2,stride=2,padding=0,bias=True)
        self.dec3 = CBR(512,256,128)

        self.unpool2 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2,padding=0,bias=True)
        self.dec2 = CBR(256,128,64)

        self.unpool1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2,padding=0,bias=True)
        self.dec1 = CBR(128,64,64)

        self.out = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0),
                            nn.Sigmoid())

    def forward(self,x):
        # constracting
        x1 = self.enc1(x) # 1->64, 256
        pool1 = self.pool(x1)

        x2 = self.enc2(pool1) # 64->128, 128
        pool2 = self.pool(x2)

        x3 = self.enc3(pool2) # 128->256, 64
        pool3 = self.pool(x3)

        x4 = self.enc4(pool3) # 256->512, 32
        pool4 = self.pool(x4)

        x5 = self.enc5(pool4) # 512->1024, 16
        
        # expanding
        y5 = self.dec5(x5) 
        unpool4 = self.unpool4(y5) # 32

        cat = torch.cat((x4,unpool4),dim=1)
        y4 = self.dec4(cat)
        unpool3 = self.unpool3(y4) # 64

        cat = torch.cat((x3,unpool3),dim=1)
        y3 = self.dec3(cat)
        unpool2 = self.unpool2(y3) # 128

        cat = torch.cat((x2,unpool2),dim=1)
        y2 = self.dec2(cat)
        unpool1 = self.unpool1(y2) # 256

        cat = torch.cat((x1,unpool1),dim=1)
        y1 = self.dec1(cat)
        
        y = self.out(y1)

        return y
