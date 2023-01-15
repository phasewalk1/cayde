"""ags.py - The PyTorch implementation of the CNN/CRNN models used in the (Adiyansjah, Gunawan, Suhartono) paper to classifiy audio segments into genres and
extract features from the audio segments to be used in the RNN model that serves recommendation purposes."""

import torch
import torch.nn as nn


"""This model is based on the paper:
    "Music Recommender System Based on Genre using Convolutional Recurrent Neural Networks": https://www.sciencedirect.com/science/article/pii/S1877050919310646
    The model is a CNN that uses binary classification to determine which genre an audio segment belongs to
"""


class BinaryClassifierCNN(nn.Module):
    # one input channel
    GRAYSCALE_INPUT_DIM = 1
    # feature maps for each convolutional layer
    CONV_FEATURE_MAPS = [(47, 95), (95, 95), (95, 142), (142, 190)]
    # num filters in convolutional layer 1
    CONV1_N_FILTERS = 47
    # num filters in convolutional layer 2 and 3
    CONV_MIDDLE_N_FILTERS = 95
    # num filters in convolutional layer 4
    CONV_MIDDLE2_N_FILTERS = 142
    # num filters in convolutional layer 5
    CONV_END_N_FILTERS = 190
    # kernel size for convolutional layers
    CONV_KERNEL_SIZE = (3, 3)
    # stride for convolutional layers
    CONV_STRIDE = 1
    # padding for all layers
    PADDING = 1
    # max-pooling kernel/stride sizes
    MAXPOOL1_KS = (2, 2)
    MAXPOOL2_KS = MAXPOOL1_KS
    MAXPOOL3_KS = MAXPOOL1_KS
    MAXPOOL4_KS = (3, 5)
    MAXPOOL5_KS = (4, 4)

    def __init__(self, output_dim):
        super(BinaryClassifierCNN, self).__init__()
        """Define the Convolutional layers"""
        # 1 input channel x 47 conv filters
        self.conv1 = nn.Conv2d(
            self.GRAYSCALE_INPUT_DIM,  # 1
            self.CONV1_N_FILTERS,  # 47
            kernel_size=self.CONV_KERNEL_SIZE,
            stride=self.CONV_STRIDE,
            padding=self.PADDING,
        )
        # 47 x 95
        self.conv2 = nn.Conv2d(
            self.CONV1_N_FILTERS,  # 47
            self.CONV_MIDDLE_N_FILTERS,  # 95
            kernel_size=self.CONV_KERNEL_SIZE,
            stride=self.CONV_STRIDE,
            padding=self.PADDING,
        )
        # 95 x 95
        self.conv3 = nn.Conv2d(
            self.CONV_MIDDLE_N_FILTERS,  # 95
            self.CONV_MIDDLE_N_FILTERS,  # 95
            kernel_size=self.CONV_KERNEL_SIZE,
            stride=self.CONV_STRIDE,
            padding=self.PADDING,
        )
        # 95 x 142
        self.conv4 = nn.Conv2d(
            self.CONV_MIDDLE_N_FILTERS,  # 95
            self.CONV_MIDDLE2_N_FILTERS,  # 142
            kernel_size=self.CONV_KERNEL_SIZE,
            stride=self.CONV_STRIDE,
            padding=self.PADDING,
        )
        # 142 x 90
        self.conv5 = nn.Conv2d(
            self.CONV_MIDDLE2_N_FILTERS,  # 142
            self.CONV_END_N_FILTERS,  # 190
            kernel_size=self.CONV_KERNEL_SIZE,
            stride=self.CONV_STRIDE,
            padding=self.PADDING,
        )

        """Define the max-pooling layers"""
        self.pool1 = nn.MaxPool2d(
            kernel_size=self.MAXPOOL1_KS,
            stride=self.MAXPOOL1_KS,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=self.MAXPOOL2_KS,
            stride=self.MAXPOOL2_KS,
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=self.MAXPOOL3_KS,
            stride=self.MAXPOOL3_KS,
        )
        self.pool4 = nn.MaxPool2d(
            kernel_size=self.MAXPOOL4_KS,
            stride=self.MAXPOOL4_KS,
        )
        self.pool5 = nn.MaxPool2d(
            kernel_size=self.MAXPOOL5_KS,
            stride=self.MAXPOOL5_KS,
        )

        """Define the batch-norm layers"""
        self.bn1 = nn.BatchNorm2d(self.CONV1_N_FILTERS)
        self.bn2 = nn.BatchNorm2d(self.CONV_MIDDLE_N_FILTERS)
        self.bn3 = nn.BatchNorm2d(self.CONV_MIDDLE_N_FILTERS)
        self.bn4 = nn.BatchNorm2d(self.CONV_MIDDLE2_N_FILTERS)
        self.bn5 = nn.BatchNorm2d(self.CONV_END_N_FILTERS)

        """Define the activation functions"""
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        """Define the fully-connected layer (output) layer"""
        self.fc = nn.Linear(self.CONV_END_N_FILTERS, output_dim)

        """Define the loss function and optimizer"""
        self.lossfn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def apply_conv_pool(self, input):
        # apply the convolutional and max-pooling layers
        conv1_out = self.relu(self.conv1(input))
        pooled1_out = self.pool1(conv1_out)
        conv2_out = self.relu(self.conv2(pooled1_out))
        pooled2_out = self.pool2(conv2_out)
        conv3_out = self.relu(self.conv3(pooled2_out))
        pooled3_out = self.pool3(conv3_out)
        conv4_out = self.relu(self.conv4(pooled3_out))
        pooled4_out = self.pool4(conv4_out)
        conv5_out = self.relu(self.conv5(pooled4_out))
        pooled5_out = self.pool5(conv5_out)

        # return the flattened output of the convolutional layers
        return pooled5_out.view(input.size(0), -1)

    def forward(self, input, targets):
        ## apply the convolutional and max-pooling layers and return the flattened output
        conv_pool_yhat = self.apply_conv_pool(input)
        ## pass into the fully-connected layer and sigmoid activation
        prediction = self.sigmoid(self.fc(conv_pool_yhat))

        ## compute the loss
        loss = self.lossfn(prediction, targets)
        ## zero the gradients
        self.optimizer.zero_grad()
        ## apply backprop
        loss.backward()
        self.optimizer.step()

        return prediction, loss


class CRNNExtractorModel(nn.Module):
    def __init__(self):
        super(CRNNExtractorModel, self).__init__()
        # Setup the convolutional/batch-norm layers
        self.conv1 = nn.Conv2d(
            in_channels=68,
            out_channels=137,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(137)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=137,
            out_channels=137,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(137)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=3)

        self.conv3 = nn.Conv2d(
            in_channels=137,
            out_channels=137,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(137)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)

        self.conv4 = nn.Conv2d(
            in_channels=137,
            out_channels=137,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(137)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)

        # Setup the GRU layers
        self.gru1 = nn.GRU(
            input_size=68,
            hidden_size=68,
            num_layers=1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=68,
            hidden_size=68,
            num_layers=1,
            batch_first=True,
        )

        # Dropout normalization
        self.dropout = nn.Dropout(p=0.1)
        # ReLU mid-layer activation
        self.relu = nn.ReLU()
        # Output layer activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # layer 1 pass
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.maxpool1(x)
        # layer 2 pass
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.maxpool2(x)
        # layer 3 pass
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.maxpool3(x)
        # layer 4 pass
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = self.maxpool4(x)

        # reshape the feature maps to be fed into the GRU
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x.squeeze(dim=1)

        # output the sigmmoid activation of the output layer
        return self.sigmoid(x)
