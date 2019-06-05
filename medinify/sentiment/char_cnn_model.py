
# Pytorch
import torch
from torch import nn
from torch.nn import functional as F


class CharCnnNet(nn.Module):
    """
    Pytorch CNN for character-based sentiment analysis
    """

    def __init__(self):
        super(CharCnnNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(70, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc3 = nn.Linear(1024, 2)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, comment):

        comment = self.conv1(comment)
        comment = self.conv2(comment)
        comment = self.conv3(comment)
        comment = self.conv4(comment)
        comment = self.conv5(comment)
        comment = self.conv6(comment)

        comment = comment.view(comment.shape[0], -1)

        comment = self.fc1(comment)
        comment = self.fc2(comment)
        comment = self.fc3(comment)
        comment = self.log_softmax(comment)
        return comment





