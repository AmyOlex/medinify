
import numpy as np

# Evaluation
from sklearn.model_selection import StratifiedKFold

# Word Embeddings
from gensim.models import Word2Vec


# Medinify
from medinify.sentiment import ReviewClassifier
from medinify.sentiment.cnn_network import SentimentNetwork

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# TorchText
from torchtext import data
from torchtext.data import Example, Dataset, Iterator
from torchtext.vocab import Vectors


class CNNReviewClassifier:
    """For performing sentiment analysis on drug reviews
        Using a PyTorch Convolutional Neural Network

    Attributes:
        vectors - TorchText word embedding vectors
        embeddings: torch tensor of word2vec embeddings
        comment_field - TorchText data field for comments
        rating_field - TorchText LabelField for ratings
        loss - CNN loss function
    """

    vectors = None
    embeddings = None
    comment_field = None
    rating_field = None
    loss = nn.BCEWithLogitsLoss()

    def __init__(self, w2v_file):
        """
        Initializes CNNReviewClassifier
        :param w2v_file: embedding file
        """
        vectors = Vectors(w2v_file)
        self.vectors = vectors

    def get_data_loaders(self, train_file, valid_file, batch_size):
        """
        Generates data_loaders given file names
        :param train_file: file with train data
        :param valid_file: file with validation data
        :param batch_size: the loaders' batch sizes
        :return: data loaders
        """

        dataset_maker = ReviewClassifier()
        train_data = dataset_maker.create_dataset(train_file)
        valid_data = dataset_maker.create_dataset(valid_file)

        return self.generate_data_loaders(train_data, valid_data, batch_size)

    def generate_data_loaders(self, train_dataset, valid_dataset, batch_size):
        """
        This function generates TorchText dataloaders for training and validation datasets
        :param train_dataset: training dataset
        :param valid_dataset: validation dataset
        :param batch_size: the loaders' batch sizes
        :return: train data loader and validation data loader
        """

        # create TorchText fields
        self.comment_field = data.Field(lower=True, dtype=torch.float64)
        self.rating_field = data.LabelField(dtype=torch.float64)

        # iterate through dataset and generate examples with comment_field and rating_field
        train_examples = []
        valid_examples = []

        for review in train_dataset:
            comment = ' '.join(list(review[0].keys()))
            rating = review[1]
            characters = list(comment)
            review = {'comment': comment, 'characters': characters, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', self.comment_field),
                                          'rating': ('rating', self.rating_field)})
            train_examples.append(ex)

        for review in valid_dataset:
            comment = ' '.join(list(review[0].keys()))
            rating = review[1]
            characters = list(comment)
            review = {'comment': comment, 'characters': characters, 'rating': rating}
            ex = Example.fromdict(data=review,
                                  fields={'comment': ('comment', self.comment_field),
                                          'rating': ('rating', self.rating_field)})
            valid_examples.append(ex)

        train_dataset = Dataset(examples=train_examples,
                                fields={'comment': self.comment_field,
                                        'rating': self.rating_field})
        valid_dataset = Dataset(examples=valid_examples,
                                fields={'comment': self.comment_field,
                                        'rating': self.rating_field})

        # build comment_field and rating_field vocabularies
        self.comment_field.build_vocab(train_dataset.comment, valid_dataset.comment,
                                       max_size=10000, vectors=self.vectors)
        self.embeddings = self.comment_field.vocab.vectors

        self.rating_field.build_vocab(['pos', 'neg'])

        # create torchtext iterators for train data and validation data
        train_loader = Iterator(train_dataset, batch_size, sort_key=lambda x: len(x))
        valid_loader = Iterator(valid_dataset, batch_size, sort_key=lambda x: len(x))

        return train_loader, valid_loader

    def batch_metrics(self, predictions, ratings):
        """
        Calculates true positive, false positive, true negative, and false negative
        given a batch's predictions and actual ratings
        :param predictions: model predictions
        :param ratings: actual ratings
        :return: number of fp, tp, tn, and fn
        """

        rounded_preds = torch.round(torch.sigmoid(predictions))

        preds = rounded_preds.to(torch.int64).numpy()
        ratings = ratings.to(torch.int64).numpy()

        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0

        i = 0
        while i < len(preds):
            if preds[i] == 0 and ratings[i] == 0:
                true_neg += 1
            elif preds[i] == 0 and ratings[i] == 1:
                false_neg += 1
            elif preds[i] == 1 and ratings[i] == 1:
                true_pos += 1
            elif preds[i] == 1 and ratings[i] == 0:
                false_pos += 1
            i += 1

        return true_pos, false_pos, true_neg, false_neg

    def train_from_files(self, train_file, valid_file, n_epochs, batch_size):
        """
        Trains a model given train file and validation file
        """

        train_loader, valid_loader = self.get_data_loaders(train_file, valid_file, batch_size)
        network = SentimentNetwork(len(self.vectors.stoi), self.vectors.vectors)
        self.train(network=network, train_loader=train_loader,
                   valid_loader=valid_loader, n_epochs=n_epochs)

    def train(self, network, train_loader, n_epochs, valid_loader=None, evaluate=True):
        """
        Trains network on training data
        :param network: network being trained
        :param train_loader: train data iterator
        :param n_epochs: number of training epochs
        :param valid_loader: validation loader
        :param evaluate: whether or not to evaluate validation set after each epoch
                (set to false during cross-validation)
        """

        optimizer = optim.Adam(network.parameters(), lr=0.001)

        network.train()

        num_epoch = 1

        for epoch in range(num_epoch, n_epochs + 1):
            print('Starting Epoch ' + str(num_epoch))

            epoch_loss = 0
            total_tp = 0
            total_fp = 0
            total_tn = 0
            total_fn = 0

            calculated = 0

            batch_num = 1
            for batch in train_loader:

                if batch_num % 25 == 0:
                    print('On batch ' + str(batch_num) + ' of ' + str(len(train_loader)))

                optimizer.zero_grad()

                # if the sentences are shorter than the largest kernel, continue to next batch
                if batch.comment.shape[0] < 4:
                    num_epoch = num_epoch + 1
                    continue

                predictions = network(batch).squeeze(1).to(torch.float64)
                tp, fp, tn, fn = self.batch_metrics(predictions, batch.rating)
                total_tp += tp
                total_tn += tn
                total_fn += fn
                total_fp += fp
                loss = self.loss(predictions, batch.rating)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                calculated = calculated + 1

                batch_num = batch_num + 1

            epoch_accuracy = (total_tp + total_tn) * 1.0 / (total_tp + total_tn + total_fp + total_fn)
            epoch_precision = total_tp * 1.0 / (total_tp + total_fp)
            epoch_recall = total_tp * 1.0 / (total_tp + total_fn)

            print('\nEpoch Loss: ' + str(epoch_loss / len(train_loader)))
            print('Epoch Accuracy: ' + str(epoch_accuracy * 100) + '%')
            print('Epoch Precision: ' + str(epoch_precision * 100) + '%')
            print('Epoch Recall: ' + str(epoch_recall * 100) + '%')
            print('True Positive: {}\tTrue Negative: {}\tFalse Positive: {}\tFalse Negative: {}\n'.format(
                total_tp, total_tn, total_fp, total_fn))

            if evaluate:
                self.evaluate(network, valid_loader)

            num_epoch = num_epoch + 1

        return network

    def evaluate(self, network, valid_loader):
        """
        Evaluates the accuracy of a model with validation data
        :param network: network being evaluated
        :param valid_loader: validation data iterator
        """

        network.eval()

        losses = []
        accuracies = []
        precisions = []
        recalls = []

        num_sample = 1

        with torch.no_grad():

            for sample in valid_loader:

                predictions = network(sample).squeeze(1)
                sample_loss = self.loss(predictions.to(torch.double),
                                        sample.rating.to(torch.double))

                tp, fp, tn, fn = self.batch_metrics(predictions, sample.rating)

                batch_accuracy = (tp + tn * 1.0) / (tp + tn + fp + fn)
                batch_precision = (tp * 1.0) / (tp + fp)
                batch_recall = (tp * 1.0) / (tp + fn)

                accuracies.append(batch_accuracy)
                precisions.append(batch_precision)
                recalls.append(batch_recall)

                losses.append(sample_loss)

                num_sample = num_sample + 1

            average_loss = np.mean(np.array(losses))
            loss_std = np.std(np.array(losses))
            average_accuracy = np.mean(np.array(accuracies)) * 100
            accuracy_std = np.std(np.array(accuracies)) * 100
            average_precision = np.mean(np.array(precisions)) * 100
            precision_std = np.std(np.array(precisions)) * 100
            average_recall = np.mean(np.array(recalls)) * 100
            recall_std = np.std(np.array(recalls)) * 100

            f1_measures = []
            for i, precision in enumerate(precisions):
                f1 = 2 * ((precision * recalls[i])/(precision + recalls[i]))
                f1_measures.append(f1)

            average_f1 = np.mean(np.array(f1_measures)) * 100
            f1_std = np.std(np.array(f1_measures)) * 100

            print('Evaluation Metrics:')
            print('Average Loss: {} +/-{}'.format(average_loss, loss_std))
            print('Average Accuracy: {}% +/-{}%'.format(average_accuracy, accuracy_std))
            print('Average Precision: {}% +/-{}%'.format(average_precision, precision_std))
            print('Average Recall: {}% +/-{}%'.format(average_recall, recall_std))
            print('Average F1 Measure: {}% +/-{}%\n'.format(average_f1, f1_std))

        return average_accuracy, average_precision, average_recall

    def set_weights(self, network):
        """
        Randomly initializes weights for neural network
        :param network: network being initialized
        :return: initialized network
        """
        if type(network) == nn.Conv2d or type(network) == nn.Linear:
            torch.nn.init.xavier_uniform_(network.weight)
            network.bias.data.fill_(0.01)

        return network

    def evaluate_k_fold(self, input_file, num_folds, num_epochs):
        """
        Evaluates CNN's accuracy using stratified k-fold validation
        :param input_file: dataset file
        :param num_folds: number of k-folds
        :param num_epochs: number of epochs per fold
        """

        classifier = ReviewClassifier()
        dataset = classifier.create_dataset(input_file)

        comments = [review[0] for review in dataset]
        ratings = [review[1] for review in dataset]

        skf = StratifiedKFold(n_splits=num_folds)

        accuracies, precisions, recalls = [], [], []

        for train, test in skf.split(comments, ratings):
            train_data = [dataset[x] for x in train]
            test_data = [dataset[x] for x in test]

            train_loader, valid_loader = self.generate_data_loaders(train_data, test_data, 25)
            network = SentimentNetwork(vocab_size=len(self.comment_field.vocab), embeddings=self.embeddings)

            network.apply(self.set_weights)

            self.train(network, train_loader, num_epochs, evaluate=False)
            fold_accuracy, fold_precision, fold_recall = self.evaluate(network, valid_loader)
            accuracies.append(fold_accuracy)
            precisions.append(fold_precision)
            recalls.append(fold_recall)

        average_accuracy = np.mean(np.array(accuracies)) * 100
        average_precision = np.mean(np.array(precisions)) * 100
        average_recall = np.mean(np.array(recalls)) * 100
        accuracy_std = np.std(np.array(accuracies)) * 100
        precision_std = np.std(np.array(precisions)) * 100
        recall_std = np.std(np.array(recalls)) * 100

        print('Average Accuracy: {}% +/-{}%'.format(average_accuracy, accuracy_std))
        print('Average Precision: {}% +/-{}%'.format(average_precision, precision_std))
        print('Average Recall: {}% +/-{}%'.format(average_recall, recall_std))

    def train_word_embeddings(self, datasets, output_file, training_epochs):
        """trains word embeddings from data files (csvs)
        Parameters:
            datasets - list of file paths to dataset csvs
            output_file - string with name of w2v file
            training_epochs - number of epochs to train embeddings
        """

        classifier = ReviewClassifier()
        comments = []
        dataset_num = 0
        for csv in datasets:
            dataset_num = dataset_num + 1
            print('Gathering comments from dataset #' + str(dataset_num))
            dataset = classifier.create_dataset(csv)
            dataset_comments = []
            for comment in dataset:
                dataset_comments.append(list(comment[0].keys()))
            print('\nFinished gathering dataset #' + str(dataset_num))
            comments = comments + dataset_comments
        print('\nGenerating Word2Vec model')
        w2v_model = Word2Vec(comments)
        print('Training word embeddings...')
        w2v_model.train(comments, total_examples=len(comments),
                        total_words=len(w2v_model.wv.vocab),
                        epochs=training_epochs)
        print('Finished training!')
        self.embeddings = w2v_model.wv
        w2v_model.wv.save_word2vec_format(output_file)

