'''
The winner of the ILSVRC 2014 competition was GoogLeNet(a.k.a. Inception V1) from Google
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        # bias is False because we have Batch Normalization
        self.conv = nn.Conv2d(in_channels,  out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self, in_channels, out1x1, out3x3_reduce, out3x3, out5x5_reduce, out5x5,  pool_proj):
        super().__init__()

        self.branch1 = Conv2dBlock(in_channels, out1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            Conv2dBlock(in_channels, out3x3_reduce, kernel_size=1),
            Conv2dBlock(out3x3_reduce, out3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            Conv2dBlock(in_channels, out5x5_reduce, kernel_size=1),
            Conv2dBlock(out5x5_reduce, out5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            # ceil_mode https://blog.csdn.net/GZHermit/article/details/79351803
            # stride=1 is necessary
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            Conv2dBlock(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], dim=1)

# used for (intermediate)output
class InceptionAux(nn.Module):

    def __init__(self, in_channels, n_classes, dropout=0.7):
        super().__init__()

        '''
        
        x = F.adaptive_avg_pool2d(x, (4, 4))
        
        '''
        self.conv = nn.Sequential(
            # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
            nn.AdaptiveAvgPool2d(output_size=(4,4)),
            # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
            Conv2dBlock(in_channels, 128, kernel_size=1)
            # N x 128 x 4 x 4
        ) 

        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x



'''
The winner of the ILSVRC 2014 competition was GoogLeNet(a.k.a. Inception V1) from Google
https://towardsdatascience.com/deep-learning-googlenet-explained-de8861c82765
https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py
'''
class GoogleNet(nn.Module):

    def __init__(self, channels = 3, n_classes=1000, aux_logits=True, 
        init_weights=True, dropout=0.2, dropout_aux=0.7, transform_input=False):

        super().__init__()
        
        self.transform_input = transform_input

        self.head = nn.Sequential(
            Conv2dBlock(channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Conv2dBlock(in_channels=64, out_channels=64, kernel_size=1),
            Conv2dBlock(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.inception3 = nn.Sequential(
            Inception( #3a
                in_channels=192, 
                out1x1=64, # branch 1
                out3x3_reduce=96, out3x3=128, #branch 2
                out5x5_reduce=16, out5x5=32, # branch 3
                pool_proj=32 # branch 4
            ),
            Inception( #3b
                in_channels=256,
                out1x1=128,
                out3x3_reduce=128, out3x3=192,
                out5x5_reduce=32, out5x5=96,
                pool_proj=64
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64) #a

        self.aux1 = None
        if aux_logits:
            self.aux1 = InceptionAux(512, n_classes, dropout=dropout_aux)

        self.inception4bcd = nn.Sequential(
            Inception(512, 160, 112, 224, 24, 64, 64), #b
            Inception(512, 128, 128, 256, 24, 64, 64), #c
            Inception(512, 112, 144, 288, 32, 64, 64), #d
        )
        
        self.aux2 = None
        if aux_logits:
            self.aux2 = InceptionAux(528, n_classes, dropout=dropout_aux)

        self.inception4e = nn.Sequential(
            Inception(528, 256, 160, 320, 32, 128, 128), #e
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )

        self.inception5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128), #a
            Inception(832, 384, 192, 384, 48, 128, 128) #b
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, n_classes)

        if init_weights:
            self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.01, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)

        x = self.head(x)
        x = self.inception3(x)
        x = self.inception4a(x)

        aux1_logits = None
        if self.training and self.aux1 is not None:
            aux1_logits = self.aux1(x)
        
        x = self.inception4bcd(x)

        aux2_logits = None
        if self.training and self.aux2 is not None:
            aux2_logits = self.aux2(x)

        x = self.inception4e(x)
        x = self.inception5(x)
        x = self.avgpool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        logits = self.fc(x)

        if aux1_logits != None and aux2_logits != None: # when aux_logits is True
            return logits, aux1_logits, aux2_logits

        return logits

if __name__ == '__main__':
    batch_img = torch.randn(size=(1, 3, 224, 224))

    googlenet = GoogleNet(n_classes=10)

    googlenet.train()

    logits, aux1_logits, aux2_logits = googlenet(batch_img)

    print(f'logits={logits}, aux1={aux1_logits}, aux2={aux2_logits}')



        



