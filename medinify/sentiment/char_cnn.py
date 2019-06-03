
# Medinify
from medinify.sentiment import CharCnnDataset, CharCnnNet

# Pytorch
import torch
from torch.utils.data import DataLoader
from torch import nn

# Evaluation
from sklearn.model_selection import StratifiedKFold

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
                if (j + 1) % 25 == 0:
                    print('On batch {} of {}'.format(j + 1, len(train_loader)))

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

            print(total_tp, total_tn, total_fp, total_fn)
            accuracy = ((total_tp + total_tn) * 1.0) / (total_tp + total_tn + total_fp + total_fn) * 100
            print('Epoch Loss: {}\nEpoch Accuracy: {}%\n'.format(running_loss, accuracy))

    def evaluate_k_fold(self, dataset, n_epochs, folds):

        comments = [review['comment'] for review in dataset.dataset]
        ratings = [review['rating'] for review in dataset.dataset]
        skf = StratifiedKFold(n_splits=folds)

        total_accuracy = 0
        total_precision = 0
        total_recall = 0

        num_fold = 1
        for train, test in skf.split(comments, ratings):
            print('Fold #' + str(num_fold))
            train_data = [dataset[x] for x in train]
            test_data = [dataset[x] for x in test]
            train_loader = self.get_data_loader(train_data, 25)
            test_loader = self.get_data_loader(test_data, 25)

            network = CharCnnNet()

            self.train(network, train_loader, n_epochs)
            fold_accuracy, fold_precision, fold_recall = self.evaluate(test_loader, network)
            total_accuracy += fold_accuracy
            total_precision += fold_precision
            total_recall += fold_recall

            num_fold += 1

        average_accuracy = total_accuracy / folds
        average_precision = total_precision / folds
        average_recall = total_recall / folds
        print('Average Accuracy: ' + str(average_accuracy))
        print('Average Precision: ' + str(average_precision))
        print('Average Recall: ' + str(average_recall))

    def evaluate(self, valid_loader, network):
        """
        Evaluates the accuracy of a model with validation data
        :param valid_loader: validation data iterator
        """
        network.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        calculated = 0

        num_sample = 1

        with torch.no_grad():

            for sample in valid_loader:

                preds = network(sample)
                loss = criterion(preds.to(torch.float64), sample['rating'].to(torch.int64))

                tp, tn, fp, fn = self.get_eval_stats(preds, sample['rating'])

                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn

                calculated += 1
                total_loss += loss

                num_sample = num_sample + 1

            average_accuracy = ((total_tp + total_tn) * 1.0 / (total_tp + total_tn + total_fp + total_fn)) * 100
            if total_tp + total_fp != 0:
                average_precision = (total_tp * 1.0 / (total_tp + total_fp)) * 100
            else:
                average_precision = 0
            average_recall = (total_tp * 1.0 / (total_tp + total_fn)) * 100
            print('Evaluation Metrics:')
            print('\nTotal Loss: {}\nAverage Accuracy: {}%\nAverage Precision: {}%\nAverage Recall: {}%'.format(
                total_loss / len(valid_loader), average_accuracy, average_precision, average_recall))
            print('True Positive: {}\tTrue Negative: {}\tFalse Positive: {}\tFalse Negative: {}\n'.format(
                total_tp, total_tn, total_fp, total_fn))

        return average_accuracy, average_precision, average_recall

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

