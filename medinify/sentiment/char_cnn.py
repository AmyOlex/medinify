
# Medinify
from medinify.sentiment import CharCnnDataset, CharCnnNet

# Pytorch
import torch
from torch.utils.data import DataLoader
from torch import nn

# Misc
import numpy as np

class CharCNN():
    """
    For training, evaluating, saving, and loading
    pytorch character-based sentiment analysis CNNs
    """

    def train(self, network, train_loader, n_epochs):

        optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for i, epoch in enumerate(range(n_epochs)):
            print('Beginning Epoch {}'.format(i + 1))
            running_loss = 0
            total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0

            for j, batch in enumerate(train_loader):

                print('On batch {} of {}'.format(j, len(train_loader)))

                optimizer.zero_grad()

                preds = network(batch)
                tp, tn, fp, fn = self.get_eval_stats(preds, batch['rating'])
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn

                loss = criterion(preds.to(torch.float64), batch['rating'].to(torch.int64))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            accuracy = ((total_tp + total_tn) * 1.0) / (total_tp + total_tn + total_fp + total_fn) * 100
            print('Epoch Loss: {}\nEpoch Accuracy: {}%\n'.format(running_loss, accuracy))

    def get_eval_stats(self, predictions, ratings):

        preds = predictions.argmax(dim=1).tolist()
        ratings = ratings.to(torch.int64).tolist()

        tp, tn, fp, fn = 0, 0, 0, 0
        for i, pred in enumerate(preds):
            if pred == 1 and ratings[i] == 1:
                tp += 1
            elif pred == 1 and ratings[i] == 0:
                fp += 1
            elif pred == 0 and ratings[i] == 0:
                tn += 1
            elif pred == 0 and ratings[i] == 1:
                fn += 1
        return tp, tn, fp, fn

    def get_data_loader(self, char_dataset, batch_size):
        """
        given a dataset, returns an iterator to feed data into network
        :param char_dataset: dataset
        :param batch_size: batch size
        :return: dataloader
        """

        return DataLoader(dataset=char_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """
        Overrides pytorch collate fn to handle variable length data
        :param batch: batch of data from dataloader
        :return: collated batch
        """

        data = np.array([review['comment'] for review in batch])
        sizes = np.array([comment.shape[0] for comment in data])
        maxlen = np.amax(sizes)
        padded_data = np.array([np.pad(comment, ((0, 1014 - comment.shape[0]), (0, 0)), 'constant', )
                                for comment in data])
        # maxlen - comment.shape[0])

        target = np.array([review['rating'] for review in batch])
        return {'comment': torch.tensor(padded_data, dtype=torch.float64),
                'rating': torch.tensor(target, dtype=torch.float64)}

