
# Medinify
from medinify.sentiment import CharCnnNet

# Pytorch
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

# Evaluation
from sklearn.model_selection import StratifiedKFold


class CharCNN():
    """
    For training, evaluating, saving, and loading
    pytorch character-based sentiment analysis CNNs
    """

    def train(self, network, train_loader, n_epochs):

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        network.train()

        num_epoch = 1
        for epoch in range(n_epochs):
            print('Beginning Epoch {}'.format(num_epoch))
            batch_num = 1
            epoch_accuracies = []
            epoch_loss = 0
            for batch in train_loader:
                comments = batch['comment']
                ratings = batch['rating']
                logit = network(comments)
                loss = F.nll_loss(logit, ratings)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
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

            num_epoch += 1

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
        criterion = nn.BCEWithLogitsLoss()

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
                loss = criterion(preds, sample['rating'])

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

    @staticmethod
    def get_data_loader(char_dataset, batch_size):

        return DataLoader(char_dataset, batch_size=batch_size)

