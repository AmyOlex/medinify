
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

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(7, 100)).to(torch.double)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(7, 256)).to(torch.double)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 256)).to(torch.double)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 256)).to(torch.double)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 256)).to(torch.double)
        self.conv6 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, 256)).to(torch.double)

        # maxpool layers
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        # fully connected layers
        self.fc1 = nn.Linear(in_features=8704, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=2)

        # softmax
        self.softmax = nn.Softmax(dim=1)

        # dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, review):

        comment = review['comment']
        # make comment shape [batch size, input channels (1), characters length, one-hot vector dim]
        comment = comment.unsqueeze(1)

        # convolve
        conv1 = F.relu(self.conv1(comment)).squeeze(3)

        # maxpool
        pooled1 = self.pool1(conv1)
        pooled1 = pooled1.unsqueeze(1).permute(0, 1, 3, 2)

        # convolve
        conv2 = F.relu(self.conv2(pooled1)).squeeze(3)

        # maxpool
        pooled2 = self.pool1(conv2).unsqueeze(1).permute(0, 1, 3, 2)

        # convolve
        conv3 = F.relu(self.conv3(pooled2)).permute(0, 3, 2, 1)
        conv4 = F.relu(self.conv4(conv3).permute(0, 3, 2, 1))
        conv5 = F.relu(self.conv5(conv4).permute(0, 3, 2, 1))
        conv6 = F.relu(self.conv6(conv5)).squeeze(3)

        # maxpool
        pooled3 = self.pool1(conv6)

        flat = torch.flatten(pooled3, start_dim=1).to(torch.float)

        # Linear layers
        linear1 = self.dropout(self.fc1(flat))
        linear2 = self.dropout(self.fc2(linear1))
        out = self.out(linear2)

        return self.softmax(out)





