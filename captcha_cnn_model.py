# -*- coding: UTF-8 -*-
import torch.nn as nn
import captcha_setting
from torchvision.models import resnet34

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.resnet = resnet34(weights='DEFAULT')
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.resnet(x)
        out = self.rfc(out)
        return out

