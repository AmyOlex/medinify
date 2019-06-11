
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

    use_c2v = False
    char_field = None
    concat = False

    def __init__(self, vocab_size, embeddings, char_field=None, concat=False, add=False):
        """
        Creates pytorch convnet for training
        :param vocab_size: size of embedding vocab
        :param embeddings: word embeddings

        """

        super(SentimentNetwork, self).__init__()

        if char_field:
            self.use_c2v = True
            self.char_field = char_field

            if not concat and not add:
                print('If using character embeddings, you must set either concat or add to true')
                exit()

            if concat:
                self.concat = True

        # embedding layers
        self.embed_words = nn.Embedding(vocab_size, 100)
        self.embed_words.weight = nn.Parameter(embeddings)

        """
        if self.use_c2v:
            self.embed_chars = nn.Embedding(char_vocab_size, 100)
            self.embed_chars.weight = nn.Parameter(char_embeddings)
        """

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

    def average_comment_embeddings(self, chars_indices, char_field, zeros_tensor):

        for num_comment, comment_indices in enumerate(chars_indices):
            words = []
            num_word = 0
            for index in comment_indices:
                if index == 2:
                    if len(words) != 0:
                        stack = torch.stack(words, dim=1)
                        average = torch.mean(stack, dim=1)
                        zeros_tensor[num_comment][num_word] = average
                        num_word += 1
                    else:
                        continue
                if index == 1:
                    break
                else:
                    words.append(char_field.vocab.vectors[index])

        zeros_tensor = zeros_tensor.unsqueeze(1)
        return zeros_tensor

    def forward(self, t):
        """
        Performs forward pass for data batch on CNN
        """

        # reshape and embed
        comments = t.comment.permute(1, 0).to(torch.long)
        embedded = self.embed_words(comments).unsqueeze(1).to(torch.double)

        if self.use_c2v:
            characters = t.characters.permute(1, 0).to(torch.long)
            zeros_tensor = torch.zeros(embedded.shape[0], embedded.shape[2], embedded.shape[3], dtype=torch.float64)
            embedded_chars = self.average_comment_embeddings(characters, self.char_field, zeros_tensor)
            if self.concat:
                embedded = torch.cat((embedded, embedded_chars), dim=2)
            else:
                embedded = torch.add(embedded, embedded_chars)

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
