
import csv

# Medinify
from medinify.sentiment import ReviewClassifier

# Pytorch
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):

    comments = []
    ratings = []
    max_len = 0
    embeddings = {}

    def __init__(self, data_file, embedding_file, max_len, use_medinify_preprocessing=False):
        """
        Initializes review dataset
        :param data_file: path for reviews file
        :param max_len: max length of review to represent (longer are truncated)
        """

        if not use_medinify_preprocessing:
            self.load_data(data_file)

        else:
            dataset_maker = ReviewClassifier()
            dataset = dataset_maker.create_dataset(data_file)
            self.comments = [' '.join(list(review[0].keys())) for review in dataset]
            self.ratings = [review[1] for review in dataset]

        self.load_embeddings(embedding_file)
        self.max_len = max_len

    def load_data(self, file):

        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['rating'] == '3':
                    continue
                if row['rating'] in ['1', '2']:
                    self.ratings.append('neg')
                elif row['rating'] in ['4', '5']:
                    self.ratings.append('pos')
                self.comments.append(row['comment'])

    def load_embeddings(self, embeddings_file):

        with open(embeddings_file, 'r') as f:
            f.readline()
            for line in f.readlines():
                split_line = line.split(' ')
                word = split_line[0]
                embedding = [float(x) for x in split_line[1:]]
                self.embeddings[word] = embedding

    def __len__(self):
        return len(self.comments)

    def encode_comment(self, index):
        comment = torch.zeros(self.max_len, 100)
        text = self.comments[index]
        for i, word in enumerate(text.split(' ')):
            if i >= self.max_len:
                break
            if word in list(self.embeddings.keys()):
                comment[i] = torch.tensor(self.embeddings[word])
        comment = torch.transpose(comment, dim0=1, dim1=0)
        return comment

    def __getitem__(self, index):
        comment = self.encode_comment(index)
        rating = torch.tensor(1)
        if self.ratings[index] == 'neg':
            rating = torch.tensor(0)
        return {'comment': comment, 'rating': rating}
