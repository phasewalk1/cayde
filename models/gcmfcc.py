"""gcmfcc.py - A PyTorch implementation of the Genre Classifier model using Mel-frequency cepstral coefficients (MFCCs) as input."""

import torch
import torch.nn as nn


"""This is another Binary Classifier, but this time using MFCCs as input instead of one-hot-encoded pixel values
from Mel-spectrograms. This is a simpler model than those implemented in the Procedia paper, and it serves a purpose
of modeling the proof of concept implementation."""


class GenreClassifierMFCC(nn.Module):
    def __init__(self, inputs):
        super(GenreClassifierMFCC, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(inputs.shape[1] * inputs.shape[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.lossfn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x, y):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))

        loss = self.lossfn(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return x, loss
