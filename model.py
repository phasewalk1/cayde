import numpy as np
import torch
import torch.nn as nn


class MelCRNN(nn.Module):
    def __init__(self):
        super(MelCRNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )

        self.fc = nn.Linear(32, 10)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, y):
        x = self.layers(x)
        # flatten the output of the conv layers
        x = x.view(x.size(0), -1)
        # apply the fully-connected layer
        x = self.fc(x)
        loss = self.loss_fn(x,y)
        # zero the gradients
        self.optimizer.zero_grad()
        # compute the gradients
        loss.backward()
        return y
