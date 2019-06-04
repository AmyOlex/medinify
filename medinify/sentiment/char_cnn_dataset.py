
# Preprocessing
import numpy, sys
from numpy import array
from sklearn.preprocessing import OneHotEncoder
from string import printable

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

    dataset = []
    onehot_encoder = None

    def __init__(self, dataset_file):
        """
        :param dataset_file: CSV dataset file
        """
        dataset = ReviewClassifier().create_dataset(dataset_file)
        for review in dataset:
            comment_text = ' '.join(list(review[0].keys()))
            if len(list(comment_text)) > 1014 or len(comment_text) < 1:
                continue

            comment_rep = self.encode_comment(comment_text)

            rating = review[1]
            rating_rep = None
            if rating == 'pos':
                rating_rep = torch.tensor(1)
            elif rating == 'neg':
                rating_rep = torch.tensor(0)

            self.dataset.append({'comment': comment_text, 'comment_rep': comment_rep,
                                 'rating': rating, 'rating_rep': rating_rep})

    def load_one_hot_encoder(self):
        """
        loads onehot_encoder for character vocabulary (ASCCI printables)
        """
        numpy.set_printoptions(threshold=sys.maxsize)

        char_vocab = [x for x in printable]
        values = array(char_vocab).reshape(len(char_vocab), 1)

        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto',
                                            handle_unknown='ignore', dtype=numpy.double)
        self.onehot_encoder.fit_transform(values)

    def encode_comment(self, comment):
        """
        Transforms string comment into one-hot encoded torch tensor
        :param comment: comment string
        :return: one-hot encoder torch tensor
        """

        torch.set_printoptions(threshold=5000)

        if not self.onehot_encoder:
            self.load_one_hot_encoder()

        characters = array(list(comment)).reshape(len(comment), 1)
        if characters.shape == (0, 1):
            print(characters)
            exit()
        encoded = self.onehot_encoder.transform(characters)
        return encoded

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        data = {'comment': sample['comment_rep'], 'rating': sample['rating_rep']}
        return data
