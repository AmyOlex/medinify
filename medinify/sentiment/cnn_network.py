
# PyTorch
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
import torch.utils.data


class SentimentNetwork(Module):
    """
    A PyTorch Convolutional Neural Network for the sentiment analysis of drug reviews
    """

    def __init__(self, vocab_size, embeddings):
        """
        Creates pytorch convnet for training
        :param vocab_size: size of embedding vocab
        :param embeddings: word embeddings

        """

        super(SentimentNetwork, self).__init__()

        # embedding layer
        self.embed_words = nn.Embedding(vocab_size, 100)
        self.embed_words.weight = nn.Parameter(embeddings)

        # convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(2, 100)).double(),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100)).double(),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 100)).double(),
            nn.ReLU()
        )

        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # fully-connected layers
        self.fc1 = nn.Linear(300, 50).float()
        self.out = nn.Linear(50, 1).float()

    def forward(self, t):
        """
        Performs forward pass for data batch on CNN
        """

        # reshape
        comments = t.comment.permute(1, 0).to(torch.long)

        # embed
        embedded = self.embed_words(comments).unsqueeze(1).to(torch.double)

        # convolve embedded outputs three times
        # to find bigrams, tri-grams, and 4-grams (or different by adjusting kernel sizes)
        convolved1 = self.conv1(embedded).squeeze(3)
        convolved2 = self.conv2(embedded).squeeze(3)
        convolved3 = self.conv3(embedded).squeeze(3)

        # maxpool convolved outputs
        pooled_1 = F.max_pool1d(convolved1, convolved1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(convolved2, convolved2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(convolved3, convolved3.shape[2]).squeeze(2)

        # concatenate maxpool outputs and dropout
        cat = self.dropout(torch.cat((pooled_1, pooled_2, pooled_3), dim=1)).to(torch.float32)

        # fully connected layers
        linear = self.fc1(cat)
        linear = F.relu(linear)
        return self.out(linear)
