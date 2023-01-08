import torch
import torch.nn as nn


class MelCRNN(nn.Module):
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

    def __init__(self, hidden_dim, output_dim, n_layers):
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

        """Define the recurrent layer"""
        self.lstm = nn.LSTM(
            self.CONV_END_N_FILTERS,
            hidden_dim,
            n_layers,
            batch_first=True,
        )

        """Define the fully-connected layer (output) layer"""
        self.fc = nn.Linear(hidden_dim, output_dim)

        """Define the loss function and optimizer"""
        self.lossfn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, input, hidden, targets):
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

        ## flatten the output of the convolutional layers
        rnn_input = pooled5_out.view(input.size(0), -1)
        ## pass into the recurrent network
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        ## pass into the fully-connected layer and sigmoid activation
        prediction = self.sigmoid(self.fc(rnn_out))
        
        ## compute the loss
        loss = self.lossfn(prediction, targets)
        ## zero the gradients
        self.optimizer.zero_grad()
        ## apply backprop
        loss.backward()
        self.optimizer.step()
        

        return prediction, hidden
