import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolutinal Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Layer 1 Convolution!
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=12, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=12, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )

        # MLP
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        # x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(self.conv4(x))

        # flatten
        x = x.view(-1, 64 * 3 * 3)

        # Classificatione
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
