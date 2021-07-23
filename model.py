from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.datasets import ImageFolder

from constants import *


class PlateClassifier(nn.Module):
    def __init__(self):
        super(PlateClassifier, self).__init__()
        self.model = resnet152(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)

