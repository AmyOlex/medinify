
# Preprocessing
import json
import csv

# Pytorch
import torch
from torch.utils.data import Dataset

# Medinify
from medinify.sentiment import ReviewClassifier


class CharCnnDataset(Dataset):
    """
    Pytorch dataset class for sentiment analysis CharCNN
    (As opposed to word level)
    """

    comments = []
    ratings = []
    alphabet = None
    max_len = 0
    processing = True

    def __init__(self, dataset_file, alphabet_file, seq_max_len, use_medinify_processing=True):
        """
        :param dataset_file: CSV dataset file
        """
        self.max_len = seq_max_len
        with open(alphabet_file, 'r') as alpha:
            self.alphabet = ''.join(json.load(alpha))
        self.processing = use_medinify_processing
        self.load_data(dataset_file)

    def load_data(self, file):

        if self.processing:
            dataset = ReviewClassifier().create_dataset(file)
            for review in dataset:
                comment = ' '.join(list(review[0].keys()))
                rating = 0
                if review[1] == 'pos':
                    rating = 1

                self.comments.append(comment)
                self.ratings.append(rating)

        else:
            with open(file, 'r') as data:
                reader = csv.DictReader(data)
                for row in reader:
                    if row['comment'] == '' or row['rating'] == 3:
                        continue
                    self.comments.append(row['comment'].lower())
                    if row['rating'] in ['1', '2']:
                        self.ratings.append(0)
                    else:
                        self.ratings.append(1)

    def encode_data(self, index):
        """
        Transforms string comment into one-hot encoded torch tensor
        :param comment: comment string
        :return: one-hot encoder torch tensor
        """

        comment_tensor = torch.zeros(len(self.alphabet), self.max_len)
        text = self.comments[index]
        for i, char in enumerate(text[::1]):
            if i >= self.max_len:
                break
            char_index = self.alphabet.find(char)
            if char_index != -1:
                comment_tensor[char_index][i] = 1.0
        return comment_tensor

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = self.encode_data(index)
        rating = self.ratings[index]
        return {'comment': comment, 'rating': rating}

