import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras


"""This model is based on the paper:
    "Music Recommender System Based on Genre using Convolutional Recurrent Neural Networks": https://www.sciencedirect.com/science/article/pii/S1877050919310646
    The model is a CNN that uses binary classification to determine which genre an audio segment belongs to
"""
class ProcediaBinaryClassifierCNN(nn.Module):
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
        super(ProcediaBinaryClassifierCNN, self).__init__()
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


"""This is another Binary Classifier, but this time using MFCCs as input. This model
also retains a higher level of simplicity since it was not derived from the paper, however,
implemented in a similar fashion to the CNN model above."""
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

    def forward(self,x,y):
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


def build_keras_genre_classifier(inputs):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile( 
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
