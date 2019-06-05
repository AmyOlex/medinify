
# Medinify
from medinify.sentiment import CharCnnNet

# Pytorch
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

# Evaluation
from sklearn.model_selection import StratifiedKFold

import os

class CharCNN():
    """
    For training, evaluating, saving, and loading
    pytorch character-based sentiment analysis CNNs
    """

    def train(self, network, train_loader, n_epochs, path, valid_loader=None):

        checkpoint = None
        epoch = 1
        if os.path.exists(path):
            checkpoint = torch.load(path)

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            network.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']

        network.train()

        for epoch in range(epoch, n_epochs + 1):
            print('Beginning Epoch {}'.format(epoch))
            batch_num = 1
            epoch_accuracies = []
            epoch_loss = 0
            for batch in train_loader:
                comments = batch['comment']
                ratings = batch['rating']
                logit = network(comments)
                batch_loss = F.nll_loss(logit, ratings)
                epoch_loss += batch_loss.item()
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                optimizer.step()

                # get the max along the 1st dim, [1] is the indices (0 or 1, pos or neg)
                if batch_num % 25 == 0:
                    print('On batch {} of {}'.format(batch_num, len(train_loader)))
                    print('Predictions:\t{}'.format(torch.max(logit, 1)[1].numpy()))
                    print('Targets:\t{}'.format(ratings.numpy()))
                    num_correct = torch.eq(ratings.to(torch.int64), torch.max(logit, 1)[1]).sum().item()
                    batch_accuracy = (num_correct * 1.0 / ratings.shape[0])
                    epoch_accuracies.append(batch_accuracy)
                    print('Batch Accuracy: {}%'.format(batch_accuracy * 100))

                batch_num += 1

            epoch_accuracy = (sum(epoch_accuracies) / len(epoch_accuracies)) * 100
            average_loss = epoch_loss / len(train_loader)
            print('\nEpoch Accuracy: {}%'.format(epoch_accuracy))
            print('Average Loss: {}'.format(average_loss))

            if valid_loader:
                self.evaluate(network, valid_loader)

            epoch += 1

            torch.save({'epoch': epoch,
                        'model_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       path)

    def evaluate(self, network, valid_loader):

        network.eval()

        total_loss = 0
        accuracies = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for batch in valid_loader:
            comments = batch['comment']
            ratings = batch['rating']
            logits = network(comments)
            loss = F.nll_loss(logits, ratings)
            total_loss += loss.item()

            preds = torch.max(logits, 1)[1]

            num_correct = torch.eq(preds, ratings).sum().item()
            batch_accuracy = num_correct * 1.0 / ratings.shape[0]
            accuracies.append(batch_accuracy)

            for i, pred in enumerate(preds):
                if pred == 1 and ratings[i] == 1:
                    total_tp += 1
                elif pred == 1 and ratings[i] == 0:
                    total_fp += 1
                elif pred == 0 and ratings[i] == 1:
                    total_fn += 1

        total_accuracy = sum(accuracies) / len(accuracies) * 100
        precision = (total_tp * 1.0) / (total_tp + total_fp) * 100
        recall = (total_tp * 1.0) / (total_tp + total_fn) * 100

        print('\nEvaluation Metrics:\n\nAverage Accuracy: {}%\nPrecision: {}%\nRecall: {}%'.
              format(total_accuracy, precision, recall))

        return total_accuracy, precision, recall

    def evaluate_k_fold(self, dataset, n_epochs, folds, path):

        comments = [review for review in dataset.comments]
        ratings = [review for review in dataset.ratings]
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

            self.train(network, train_loader, n_epochs, path)
            fold_accuracy, fold_precision, fold_recall = self.evaluate(network, test_loader)
            total_accuracy += fold_accuracy
            total_precision += fold_precision
            total_recall += fold_recall

            with open(path, 'wb') as f:
                info = torch.load(f)
                print(info)
                exit()

            num_fold += 1

        average_accuracy = total_accuracy / folds
        average_precision = total_precision / folds
        average_recall = total_recall / folds
        print('Average Accuracy: ' + str(average_accuracy) + '%')
        print('Average Precision: ' + str(average_precision) + '%')
        print('Average Recall: ' + str(average_recall) + '%')

    @staticmethod
    def get_data_loader(char_dataset, batch_size):

        return DataLoader(char_dataset, batch_size=batch_size)

