'''
https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
'''

import torch
import torch.nn as nn
from torch.nn.modules import padding

class AlexNetOriginal(nn.Module):
    def __init__(self, n_classes=1000, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            #input: [224, 224, 3]， output: [55, 55, 48]
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2), 
            nn.ReLU(inplace=True),
            #input: [55, 55, 48]， output: [27, 27, 48]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            #input：[27, 27, 48]， output: [27, 27, 128]
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # input: [27, 27, 128]，output: [13, 13, 128]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # input: [13, 13, 128]， output: [13, 13, 192]
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # input: [13, 13, 192]， output: [13, 13, 192]
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # input: [13, 13, 192]， output: [13, 13, 128]
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # input: [13, 13, 128]， output: [6, 6, 128]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(6*6*128, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, n_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        feature = self.features(x)
        feature_flatten = torch.flatten(feature, start_dim=1)
        logits = self.classifier(feature_flatten)

        probs = torch.softmax(logits, dim=1)

        return logits, probs

'''
https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
'''
class AlexNet2(nn.Module):
    def __init__(self, n_classes=1000, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6,6)) # [256, 6, 6]
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=n_classes)
        )

    def forward(self, x):
        feature = self.features(x)
        feature = self.avgpool(feature)
        feature_flatten = torch.flatten(feature, start_dim=1)
        logits = self.classifier(feature_flatten)

        probs = torch.softmax(logits, dim=1)

        return logits, probs


if __name__ == '__main__':
    tensor_image = torch.randn(size=(3, 224, 224))
    alexnet1 = AlexNetOriginal(n_classes=10)

    logits, probs = alexnet1(tensor_image.unsqueeze(0))
    print(f'AlexNet1, logits={logits}, probs={probs}')

    alexnet2 = AlexNet2(n_classes=10)
    logits, probs = alexnet2(tensor_image.unsqueeze(0))
    print(f'AlexNet2, logits={logits}, probs={probs}')

