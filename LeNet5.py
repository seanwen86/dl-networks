'''
https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes)
        )

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature_flatten = torch.flatten(feature, start_dim=1)
        logits = self.classifier(feature_flatten)
        probs = F.softmax(logits, dim=1)

        return logits, probs

if __name__ == '__main__':
    
    gray_image = torch.randn(size=(1,32,32))
    
    lenet5 = LeNet5(2)

    batch_images = gray_image.unsqueeze(0)
    logits, probs = lenet5(batch_images)

    print(logits, probs)

