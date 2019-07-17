
# Python Libraries
import pickle
import tarfile
import os
import string
import itertools
import time

# Preprocessings
import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Classification
from sklearn import svm

# Evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# NN (Currently Unused)
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout


class ReviewClassifier:
    """
    This class is used for the training and evaluation of supervised machine learning classifiers,
    currently including Multinomial Naive Bayes, Random Forest, and Support Vector Machine (all
    implemented using the SciKit Learn library) for the sentiment analysis of online drug reviews
    Attributes:
        classifier_type: str
            acronym for the type of machine learning algorithm used
            ('nb' - Multinomial Naive Bayes, 'rf' - Random Forest, 'svm' - Support Vector Machine)
        model: MultinomialNaiveBayes, RandomForestClassifier, or LinearSVC (depending on classifier type)
            an instance's trained or training classification model
        negative_threshold: float
            star-rating cutoff at with anything <= is labelled negative (default 2.0)
        positive_threshold: float
            star-rating cutoff at with anything >= is labelled positive (default 4.0)
        vectorizer: DictVectorizer
            object for turning dictionary of tokens into numerical representation (vector)
    """

    classifier_type = None
    model = None
    numclasses = 2
    negative_threshold = 2.0
    positive_threshold = 4.0
    vectorizer = None

    def __init__(self, classifier_type=None, numclasses=2, negative_threshold=None, positive_threshold=None):
        """
        Initialize an instance of ReviewClassifier for the processing of review data into numerical
        representations, training machine-learning classifiers, and evaluating these classifiers' effectiveness
        :param classifier_type: SciKit Learn supervised machine-learning classifier ('nb', 'svm', or 'rf')
        :param negative_threshold: star-rating cutoff at with anything <= is labelled negative (default 2.0)
        :param positive_threshold: star-rating cutoff at with anything >= is labelled positive (default 4.0)
        """

        self.classifier_type = classifier_type
        self.vectorizer = DictVectorizer(sparse=False)
        self.numclasses = numclasses

        if negative_threshold:
            self.negative_threshold = negative_threshold
        if positive_threshold:
            self.positive_threshold = positive_threshold

    def preprocess(self, reviews_filename, count=False):
        """
        Transforms reviews (comments and ratings) into numerical representations (vectors)
        Comments are vectorized into bag-of-words representation
        Ratings are transformed into 0's (negative) and 1's (positive)
        Neutral reviews are discarded
        :param reviews_filename: CSV file with comments and ratings
        :return:
        data: list of sparse matrices
            vectorized comments
        target: list of integers
            vectorized ratings
        """

        stop_words = set(stopwords.words('english'))
        sp = spacy.load('en_core_web_sm')

        df = pd.read_csv(reviews_filename)
        raw_data, raw_target = [], []

        for review in df.itertuples():

            if type(review.comment) == float:
                continue

            if not count:
                comment = {token.text: True for token in sp.tokenizer(review.comment.lower()) if token.text
                           not in stop_words and not token.is_punct and not token.is_space}

                if self.numclasses == 2:
                    rating = 'pos'
                    if review.rating == 3:
                        continue
                    if review.rating in [1, 2]:
                        rating = 'neg'
                    raw_data.append(comment)
                    raw_target.append(rating)

                elif self.numclasses == 3:
                    rating = 'neg'
                    if review.rating == 3:
                        rating = 'neut'
                    elif review.rating in [4, 5]:
                        rating = 'pos'
                    raw_data.append(comment)
                    raw_target.append(rating)

                elif self.numclasses == 5:
                    raw_target.append(review.rating)
                    raw_data.append(comment)

            else:

                if review.rating == 3:
                    continue
                comment = ' '.join([token.text for token in sp.tokenizer(review.comment) if token.text
                                    not in stop_words and not token.is_punct and not token.is_space])
                rating = 'pos'
                if review.rating in [1, 2]:
                    rating = 'neg'
                raw_data.append(comment)
                raw_target.append(rating)

        encoder = LabelEncoder()
        target = np.asarray(encoder.fit_transform(raw_target))

        if not count:
            data = self.vectorizer.fit_transform(raw_data)
        else:
            data = np.asarray([x.todense() for x in CountVectorizer().fit_transform(raw_data)]).squeeze(1)

        return data, target

    def generate_model(self):
        """
        Creates model based on classifier type
        :return model: untrained machine learning classifier
        """

        model = None

        if self.classifier_type == 'nb':
            model = MultinomialNB(alpha=1, fit_prior=True)
        elif self.classifier_type == 'rf':
            model = RandomForestClassifier(n_estimators=100)
        elif self.classifier_type == 'svm':
            model = svm.LinearSVC(max_iter=10000)

        return model

    def fit(self, data, target):
        """
        Fits model to data and targets
        :param data: list of vectorized comments
        :param target: assosiated ratings (0's and 1's)
        :return model: trained machine learning classifier
        """

        model = self.generate_model()
        model.fit(data, target)
        self.model = model
        return model

    def evaluate_accuracy(self, data, target, model=None):
        """Evaluate accuracy of current model on new data
        Args:
            data: vectorized comments for feed into model
            target: actual ratings assosiated with data
            model: trained model to evaluate (if none, the class attribute 'model' will be evaluated)
        """

        if model:
            predictions = model.predict(data)

        else:
            predictions = self.model.predict(data)

        results = self.metrics(target, predictions)

        if self.numclasses == 2:
            print('Evaluation Metrics:')
            print('Accuracy: {}%'.format(results['accuracy'] * 100))
            print('Positive Precision: {}%'.format(results['precision1'] * 100))
            print('Positive Recall: {}%'.format(results['recall1'] * 100))
            print('Positive F1-Score: {}%'.format(results['f1_1'] * 100))
            print('Negative Precision: {}%'.format(results['precision2'] * 100))
            print('Negative Recall: {}%'.format(results['recall2'] * 100))
            print('Negative F1-Score: {}%'.format(results['f1_2'] * 100))

        if self.numclasses == 3:
            print('Evaluation Metrics:')
            print('Accuracy: {}%'.format(results['accuracy'] * 100))
            print('Positive Precision: {}%'.format(results['precision1'] * 100))
            print('Positive Recall: {}%'.format(results['recall1'] * 100))
            print('Positive F1-Score: {}%'.format(results['f1_1'] * 100))
            print('Negative Precision: {}%'.format(results['precision2'] * 100))
            print('Negative Recall: {}%'.format(results['recall2'] * 100))
            print('Negative F1-Score: {}%'.format(results['f1_2'] * 100))
            print('Neutral Precision: {}%'.format(results['precision3'] * 100))
            print('Neutral Recall: {}%'.format(results['recall3'] * 100))
            print('Neutral F1-Score: {}%'.format(results['f1_3'] * 100))

        if self.numclasses == 5:
            print('Evaluation Metrics:')
            print('Accuracy: {}%'.format(results['accuracy'] * 100))
            print('One Star Precision: {}%'.format(results['precision1'] * 100))
            print('One Star Recall: {}%'.format(results['recall1'] * 100))
            print('One Star F1-Score: {}%'.format(results['f1_1'] * 100))
            print('Two Star Precision: {}%'.format(results['precision2'] * 100))
            print('Two Star Recall: {}%'.format(results['recall2'] * 100))
            print('Two Star F1-Score: {}%'.format(results['f1_2'] * 100))
            print('Three Star Precision: {}%'.format(results['precision3'] * 100))
            print('Three Star Recall: {}%'.format(results['recall3'] * 100))
            print('Three Star F1-Score: {}%'.format(results['f1_3'] * 100))
            print('Four Star Precision: {}%'.format(results['precision4'] * 100))
            print('Four Star Recall: {}%'.format(results['recall4'] * 100))
            print('Four Star F1-Score: {}%'.format(results['f1_4'] * 100))
            print('Five Star Precision: {}%'.format(results['precision5'] * 100))
            print('Five Star Recall: {}%'.format(results['recall5'] * 100))
            print('Five Star F1-Score: {}%'.format(results['f1_5'] * 100))

        """
        if self.classifier_type == 'nn':
            score = self.model.evaluate(
                test_data, np.array(test_target), verbose=0)[1]
        """

        return results

    def evaluate_average_accuracy(self, reviews_filename, n_folds, count=False):
        """ Use stratified k fold to calculate average accuracy of models
        Args:
            reviews_filename: Filename of CSV with reviews to train on
            n_folds: int, number of k-folds
        """

        data, target = self.preprocess(reviews_filename=reviews_filename, count=count)
        splits = StratifiedKFold(n_splits=n_folds)

        """
        total_tn, total_fp, sumfn, sumtp = 0, 0, 0, 0
        accuracies, class_1_precisions, class_1_recalls, class_1_f1s = [], [], [], []
        class_2_precisions, class_2_recalls, class_2_f1s = [], [], []
        class_3_precisions, class_3_recalls, class_3_f1s = [], [], []
        class_4_precisions, class_4_recalls, class_4_f1s = [], [], []
        class_5_precisions, class_5_recalls, class_5_f1s = [], [], []
        sumtpPos = sumtpNeg = sumtpNeu = sumtpOneStar = sumtpTwoStar = sumtpThreeStar = sumtpFourStar = \
            sumtpFiveStar = sumfAB = sumfAC = sumfAD = sumfAE = sumfBA = sumfBC = sumfBD = sumfBE = \
            sumfCA = sumfCB = sumfCD = sumfCE = sumfDA = sumfDB = sumfDC = sumfDE = sumfEA = sumfEB = \
            sumfEC = sumfED = 0
        """

        for train, test in splits.split(data, target):

            x_train = np.array([data[x] for x in train])
            y_train = np.array([target[x] for x in train])
            x_test = np.array([data[x] for x in test])
            y_test = np.array([target[x] for x in test])

            print(x_train.shape)

            model = self.generate_model()
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            matrix = confusion_matrix(y_test, preds)
            accuracy = accuracy_score(y_test, preds)
            tp, tn, fp, fn = matrix[1][1], matrix[0][0], matrix[0][1], matrix[1][0]
            pos_precision = (tp * 1.0) / (tp + fp)
            pos_recall = (tp * 1.0) / (tp + fn)
            neg_precision = (tn * 1.0) / (tn + fn)
            neg_recall = (tn * 1.0) / (tn + fp)
            print('Fold Metrics:')
            print('Accuracy: {}%'.format(accuracy * 100))
            print('Positive Precision: {}%'.format(pos_precision * 100))
            print('Positive Recall: {}%'.format(pos_recall * 100))
            print('Negative Precision: {}%'.format(neg_precision * 100))
            print('Negative Recall: {}%'.format(neg_recall * 100))

        """
        results = self.evaluate_accuracy(x_test, y_test, model=model)
        accuracies.append(results['accuracy'])
        class_1_precisions.append(results['precision1'])
        class_2_precisions.append(results['precision2'])
        class_1_recalls.append(results['recall1'])
        class_2_recalls.append(results['recall2'])
        class_1_f1s.append(results['f1_1'])
        class_2_f1s.append(results['f1_2'])
        if self.numclasses == 2:
            total_tn += results['tn']
            sumfn += results['fn']
            total_fp += results['fp']
            sumtp += results['tp']
        if self.numclasses == 3:
            sumtpPos += results['tp_pos']
            sumtpNeg += results['tp_neg']
            sumtpNeu += results['tp_neut']
            sumfBA += results['f_ba']
            sumfBC += results['f_bc']
            sumfAB += results['f_ab']
            sumfCB += results['f_cb']
            sumfCA += results['f_ca']
            sumfAC += results['f_ac']
        if self.numclasses == 5:
            sumtpOneStar += results['tp_one']
            sumtpTwoStar += results['tp_two']
            sumtpThreeStar += results['tp_three']
            sumtpFourStar += results['tp_four']
            sumtpFiveStar += results['tp_five']
            sumfAB += results['f_ab']
            sumfAC += results['f_ac']
            sumfAD += results['f_ad']
            sumfAE += results['f_ae']
            sumfBA += results['f_ba']
            sumfBC += results['f_bc']
            sumfBD += results['f_bd']
            sumfBE += results['f_be']
            sumfCA += results['f_ca']
            sumfCB += results['f_cb']
            sumfCD += results['f_cd']
            sumfCE += results['f_ce']
            sumfDA += results['f_da']
            sumfDB += results['f_db']
            sumfDC += results['f_dc']
            sumfDE += results['f_de']
            sumfEA += results['f_ea']
            sumfEB += results['f_eb']
            sumfEC += results['f_ec']
            sumfED += results['f_ed']
        if self.numclasses == 3 or self.numclasses == 5:
            class_3_precisions.append(results['precision3'])
            class_3_recalls.append(results['recall3'])
            class_3_f1s.append(results['f1_3'])
        if self.numclasses == 5:
            class_4_precisions.append(results['precision4'])
            class_4_recalls.append(results['recall4'])
            class_4_f1s.append(results['f1_4'])
            class_5_precisions.append(results['precision5'])
            class_5_recalls.append(results['recall5'])
            class_5_f1s.append(results['f1_5'])
        average_accuracy = np.mean(np.array(accuracies)) * 100
        accuracy_std = np.std(np.array(accuracies)) * 100
        average_precision1 = np.mean(np.array(class_1_precisions)) * 100
        precision1_std = np.std(np.array(class_1_precisions)) * 100
        average_precision2 = np.mean(np.array(class_2_precisions)) * 100
        precision2_std = np.std(np.array(class_2_precisions)) * 100
        average_recall1 = np.mean(np.array(class_1_recalls)) * 100
        recall1_std = np.std(np.array(class_1_recalls)) * 100
        average_recall2 = np.mean(np.array(class_2_recalls)) * 100
        recall2_std = np.std(np.array(class_2_recalls)) * 100
        average_f1_1 = np.mean(np.array(class_1_f1s)) * 100
        f1_1_std = np.std(np.array(class_1_f1s)) * 100
        average_f1_2 = np.mean(np.array(class_2_f1s)) * 100
        f1_2_std = np.std(np.array(class_2_f1s)) * 100
        if self.numclasses == 3 or self.numclasses == 5:
            average_precision3 = np.mean(np.array(class_3_precisions)) * 100
            precision3_std = np.std(np.array(class_3_precisions)) * 100
            average_recall3 = np.mean(np.array(class_3_recalls)) * 100
            recall3_std = np.std(np.array(class_3_recalls)) * 100
            average_f1_3 = np.mean(np.array(class_3_f1s)) * 100
            f1_3_std = np.std(np.array(class_3_f1s)) * 100
        if self.numclasses == 5:
            average_precision4 = np.mean(np.array(class_4_precisions)) * 100
            precision4_std = np.std(np.array(class_4_precisions)) * 100
            average_recall4 = np.mean(np.array(class_4_recalls)) * 100
            recall4_std = np.std(np.array(class_4_recalls)) * 100
            average_f1_4 = np.mean(np.array(class_4_f1s)) * 100
            f1_4_std = np.std(np.array(class_4_f1s)) * 100
            average_precision5 = np.mean(np.array(class_5_precisions)) * 100
            precision5_std = np.std(np.array(class_5_precisions)) * 100
            average_recall5 = np.mean(np.array(class_5_recalls)) * 100
            recall5_std = np.std(np.array(class_5_recalls)) * 100
            average_f1_5 = np.mean(np.array(class_5_f1s)) * 100
            f1_5_std = np.std(np.array(class_5_f1s)) * 100
        if self.numclasses == 2:
            print('Validation Metrics:')
            print('Average Accuracy: {:.4f}% +/- {:.4f}%'.format(average_accuracy, accuracy_std))
            print('Average Precision: {:.4f}% +/- {:.4f}%'.format((average_precision1 + average_precision2) / 2, (precision1_std + precision2_std) / 2))
            print('Average Recall: {:.4f}% +/- {:.4f}%'.format((average_recall1 + average_recall2) / 2, (recall1_std + recall2_std) / 2))
            print('Average F1-Score: {:.4f}% +/- {:.4f}%'.format((average_f1_1 + average_f1_2) / 2, (f1_1_std + f1_2_std) / 2))
            print('Average Class 1 (Positive) Precision: {:.4f}% +/- {:.4f}%'.format(average_precision1, precision1_std))
            print('Average Class 1 (Positive) Recall: {:.4f}% +/- {:.4f}%'.format(average_recall1, recall1_std))
            print('Average Class 1 (Positive) F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_1, f1_1_std))
            print('Average Class 2 (Negative) Precision: {:.4f}% +/- {:.4f}%'.format(average_precision2, precision2_std))
            print('Average Class 2 (Negative) Recall: {:.4f}% +/- {:.4f}%'.format(average_recall2, recall2_std))
            print('Average Class 2 (Negative) F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_2, f1_2_std))
            print('Confusion Matrix: ')
            print("\t" + "\t" + "Neg:" + "\t" + "Pos:")
            print("Negative:" + "\t" + str(total_tn) + "\t" + str(total_fp))
            print("Positive:" + "\t" + str(sumfn) + "\t" + str(sumtp))
        elif self.numclasses == 3:
            print('Validation Metrics:')
            print('Average Accuracy: {:.4f}% +/- {:.4f}%'.format(average_accuracy, accuracy_std))
            print('Average Precision: {:.4f}% +/- {:.4f}%'.format((average_precision1 + average_precision2 + average_precision3) / 3, (precision1_std + precision2_std + precision3_std) / 3))
            print('Average Recall: {:.4f}% +/- {:.4f}%'.format((average_recall1 + average_recall2 + average_recall3) / 3, (recall1_std + recall2_std + recall3_std) / 3))
            print('Average F1-Score: {:.4f}% +/- {:.4f}%'.format((average_f1_1 + average_f1_2 + average_f1_3) / 3, (f1_1_std + f1_2_std + f1_3_std) / 3))
            print('Average Class 1 (Positive) Precision: {:.4f}% +/- {:.4f}%'.format(average_precision1, precision1_std))
            print('Average Class 1 (Positive) Recall: {:.4f}% +/- {:.4f}%'.format(average_recall1, recall1_std))
            print('Average Class 1 (Positive) F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_1, f1_1_std))
            print('Average Class 2 (Negative) Precision: {:.4f}% +/- {:.4f}%'.format(average_precision2, precision2_std))
            print('Average Class 2 (Negative) Recall: {:.4f}% +/- {:.4f}%'.format(average_recall2, recall2_std))
            print('Average Class 2 (Negative) F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_2, f1_2_std))
            print('Average Class 3 (Neutral) Precision: {:.4f}% +/- {:.4f}%'.format(average_precision3, precision3_std))
            print('Average Class 3 (Neutral) Recall: {:.4f}% +/- {:.4f}%'.format(average_recall3, recall3_std))
            print('Average Class 3 (Neutral) F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_3, f1_3_std))
            print('Confusion Matrix: ')
            print("\t" + "\t" + "Neg:" + "\t" + "Neu:" + "\t" + "Pos:")
            print("Negative:" + "\t" + str(sumtpPos) + "\t" + str(sumfAB) + "\t" + str(sumfAC))
            print("Neutral:" + "\t" + str(sumfBA) + "\t" + str(sumtpNeg) + "\t" + str(sumfBC))
            print("Positive:" + "\t" + str(sumfCA) + "\t" + str(sumfCB) + "\t" + str(sumtpNeu))
        elif self.numclasses == 5:
            print('Validation Metrics:')
            print('Average Accuracy: {:.4f}% +/- {:.4f}%'.format(average_accuracy, accuracy_std))
            print('Average Precision: {:.4f}% +/- {:.4f}%'.format((average_precision1 + average_precision2 + average_precision3 + average_precision4 + average_precision5) / 5, (precision1_std + precision2_std + precision3_std + precision4_std + precision5_std) / 5))
            print('Average Recall: {:.4f}% +/- {:.4f}%'.format((average_recall1 + average_recall2 + average_recall3 + average_recall4 + average_recall5) / 5, (recall1_std + recall2_std + recall3_std + recall4_std + recall5_std) / 5))
            print('Average F1-Score: {:.4f}% +/- {:.4f}%'.format((average_f1_1 + average_f1_2 + average_f1_3 + average_f1_4 + average_f1_5) / 5, (f1_1_std + f1_2_std + f1_3_std + f1_4_std + f1_5_std) / 5))
            print('Average One Star Precision: {:.4f}% +/- {:.4f}%'.format(average_precision1, precision1_std))
            print('Average One Star Recall: {:.4f}% +/- {:.4f}%'.format(average_recall1, recall1_std))
            print('Average One Star F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_1, f1_1_std))
            print('Average Two Star Precision: {:.4f}% +/- {:.4f}%'.format(average_precision2, precision2_std))
            print('Average Two Star Recall: {:.4f}% +/- {:.4f}%'.format(average_recall2, recall2_std))
            print('Average Two Star F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_2, f1_2_std))
            print('Average Three Star Precision: {:.4f}% +/- {:.4f}%'.format(average_precision3, precision3_std))
            print('Average Three Star Recall: {:.4f}% +/- {:.4f}%'.format(average_recall3, recall3_std))
            print('Average Three Star F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_3, f1_3_std))
            print('Average Four Star Precision: {:.4f}% +/- {:.4f}%'.format(average_precision4, precision4_std))
            print('Average Four Star Recall: {:.4f}% +/- {:.4f}%'.format(average_recall4, recall4_std))
            print('Average Four Star F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_4, f1_4_std))
            print('Average Five Star Precision: {:.4f}% +/- {:.4f}%'.format(average_precision5, precision5_std))
            print('Average Five Star Recall: {:.4f}% +/- {:.4f}%'.format(average_recall5, recall5_std))
            print('Average Five Star F1-Score: {:.4f}% +/- {:.4f}%'.format(average_f1_5, f1_5_std))
            print('Confusion Matrix: ')
            print("\t" + "\t" + "1-Star:" + "\t" + "2-Star:" + "\t" + "3-Star:" + "\t" + "4-Star:" + "\t" + "5-Star:")
            print("One Star:" + "\t" + str(sumtpOneStar) + "\t" + str(sumfAB) + "\t" + str(sumfAC) + "\t" + str(sumfAD) + "\t" + str(sumfAE))
            print("Two Star:" + "\t" + str(sumfBA) + "\t" + str(sumtpTwoStar) + "\t" + str(sumfBC) + "\t" + str(sumfBD) + "\t" + str(sumfBE))
            print("Three Star:" + "\t" + str(sumfCA) + "\t" + str(sumfCB) + "\t" + str(sumtpThreeStar) + "\t" + str(sumfCD) + "\t" + str(sumfCE))
            print("Four Star:" + "\t" + str(sumfDA) + "\t" + str(sumfDB) + "\t" + str(sumfDC) + "\t" + str(sumtpFourStar) + "\t" + str(sumfDE))
            print("Five Star:" + "\t" + str(sumfEA) + "\t" + str(sumfEB) + "\t" + str(sumfEC) + "\t" + str(sumfED) + "\t" + str(sumtpFiveStar))
            """

    def classify(self, output_file, csv_file=None, text_file=None, evaluate=False):
        """Classifies a list of comments as positive or negative
        Args:
            output_file: txt file to which classification results will output
            csv_file: CSV file with comments to classify
            text_file: txt file with comments and no ratings
            evaluate: whether or not to write evaluation metrics to output file
        """

        if self.model is None:
            raise Exception('Model needs training first')
        if self.model and not self.vectorizer:
            raise Exception('A model must be trained before classifying')
        if text_file and evaluate:
            raise Exception('In order to evaluate the classification, data must be passed in csv format')

        df = pd.read_csv(csv_file)
        comments = []

        for review in df.itertuples():
            if type(review.comment) == float:
                continue

            if self.numclasses == 2:
                if review.rating == 3:
                    continue
                comments.append(review.comment)

            else:
                comments.append(review.comment)

        data, target = self.preprocess(csv_file)
        predictions = self.model.predict(data)

        classifications_file = open(output_file, 'a')
        if self.numclasses == 2:
            for i, comment in enumerate(comments):
                if predictions[i] == 1:
                    pred = 'Positive'
                else:
                    pred = 'Negative'
                if target[i] == 0:
                    actual = 'Negative'
                else:
                    actual = 'Positive'
                classifications_file.write('Comment: {}\tPrediction: {}\tActual Rating: {}\n'.format(comment, pred,
                                                                                                     actual))

        if self.numclasses == 3:
            for i, comment in enumerate(comments):
                if predictions[i] == 2:
                    pred = 'Positive'
                elif predictions[i] == 1:
                    pred = 'Neutral'
                else:
                    pred = 'Negative'
                if target[i] == 0:
                    actual = 'Negative'
                elif target[i] == 1:
                    actual = 'Neutral'
                else:
                    actual = 'Positive'
                classifications_file.write('Comment: {}\tPrediction: {}\tActual Rating: {}\n'.format(comment, pred,
                                                                                                     actual))
        if self.numclasses == 5:
            for i, comment in enumerate(comments):
                if predictions[i] == 1:
                    pred = 'One Star'
                elif predictions[i] == 2:
                    pred = 'Two Star'
                elif predictions[i] == 3:
                    pred = 'Three Star'
                elif predictions[i] == 4:
                    pred = 'Four Star'
                else:
                    pred = 'Five Star'
                if target[i] == 1:
                    actual = 'One Star'
                elif target[i] == 2:
                    actual = 'Two Star'
                elif target[i] == 3:
                    actual = 'Three Star'
                elif target[i] == 4:
                    actual = 'Four Star'
                else:
                    actual = 'Five Star'
                classifications_file.write('Comment: {}\tPrediction: {}\tActual Rating: {}\n'.format(comment, pred,
                                                                                                     actual))

        if evaluate:
            results = self.metrics(target, predictions)
            classifications_file.write('\nEvaluation Metrics:\n')
            if self.numclasses == 2:
                classifications_file.write(
                    'Accuracy: {}%\nClass 1 (Positive) Precision: {}%\n'
                    'Class 1 (Positive) Recall: {}%\nClass 1 (Positive) F1-Measure: {}%\n'
                    'Class 2 (Negative) Precision: {}%\nClass 2 (Negative) Recall: {}%\n'
                    'Class 2 (Negative) F1-Measure: {}%'.format(results['accuracy'] * 100,
                                                                results['precision1'] * 100,
                                                                results['recall1'] * 100,
                                                                results['f1_1'] * 100,
                                                                results['precision2'] * 100,
                                                                results['recall2'] * 100,
                                                                results['f1_2'] * 100))
            elif self.numclasses == 3:
                classifications_file.write(
                    'Accuracy: {}%\nClass 1 (Positive) Precision: {}%\n'
                    'Class 1 (Positive) Recall: {}%\nClass 1 (Positive) F1-Measure: {}%\n'
                    'Class 2 (Negative) Precision: {}%\nClass 2 (Negative) Recall: {}%\n'
                    'Class 2 (Negative) F1-Measure: {}%\nClass 3 (Neutral) Precision: {}%\n'
                    'Class 3 (Neutral) Recall: {}%\n'
                    'Class 3 (Neutral) F1-Measure: {}%\n'.format(results['accuracy'] * 100,
                                                                 results['precision1'] * 100,
                                                                 results['recall1'] * 100,
                                                                 results['f1_1'] * 100,
                                                                 results['precision2'] * 100,
                                                                 results['recall2'] * 100,
                                                                 results['f1_2'] * 100,
                                                                 results['precision3'] * 100,
                                                                 results['recall3'] * 100,
                                                                 results['f1_3'] * 100))
            elif self.numclasses == 5:
                classifications_file.write(
                    'Accuracy: {}%\nOne Star Precision: {}%\n'
                    'One Star Recall: {}%\nOne Star F1-Measure: {}%\n'
                    'Two Star Precision: {}%\nTwo Star Recall: {}%\n'
                    'Two Star F1-Measure: {}%\nThree Star Precision: {}%\n'
                    'Three Star Recall: {}%\nThree Star F1-Measure: {}%\n'
                    'Four Star Precision: {}%\nFour Star Recall: {}%\n'
                    'Four Star F1-Measure: {}%\nFive Star Precision: {}%\n'
                    'Five Star Recall: {}%\nFive Star F1-Measure: {}%\n'.format(
                        results['accuracy'] * 100, results['precision1'] * 100,
                        results['recall1'] * 100, results['f1_1'] * 100,
                        results['precision2'] * 100, results['recall2'] * 100,
                        results['f1_2'] * 100, results['precision3'] * 100,
                        results['recall3'] * 100, results['f1_3'] * 100,
                        results['precision4'] * 100, results['recall4'] * 100,
                        results['f1_4'] * 100, results['precision5'] * 100,
                        results['recall5'] * 100, results['f1_5'] * 100))

    def save_model(self, output_file):
        """ Saves a trained model to a file
        """

        if self.classifier_type and self.classifier_type != 'nn':
            with open(output_file, 'wb') as pickle_file:
                pickle.dump(self.model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.vectorizer, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        elif self.classifier_type == 'nn':
            with open("trained_nn_model.json", "w") as json_file:
                json_file.write(self.model.to_json()) # Save mode
            self.model.save_weights("trained_nn_weights.h5") # Save weights
            with open('trained_nn_vec_encoder.pickle', 'wb') as pickle_file:
                pickle.dump(self.vectorizer, pickle_file)
                # pickle.dump(self.encoder, pickle_file)
            tar_file = tarfile.open("trained_nn_model.tar", 'w')
            tar_file.add('trained_nn_model.json')
            tar_file.add('trained_nn_weights.h5')
            tar_file.add('trained_nn_vec_encoder.pickle')
            tar_file.close()

            os.remove('trained_nn_model.json')
            os.remove('trained_nn_weights.h5')
            os.remove('trained_nn_vec_encoder.pickle')

    def load_model(self, model_file=None, tar_file=None, saved_vectorizer=None):
        """ Loads a trained model from a file
        """

        with open(model_file, 'rb') as model_file:
            self.model = pickle.load(model_file)
            self.vectorizer = pickle.load(model_file)

        if saved_vectorizer and tar_file:
            tfile = tarfile.open(tar_file, 'r')
            tfile.extractall()
            tfile.close()

            with open('trained_nn_model.json', 'r') as json_model:
                loaded_model = json_model.read()
                self.model = model_from_json(loaded_model)

            self.model.load_weights('trained_nn_weights.h5')
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            with open('trained_nn_vec_encoder.pickle', 'rb') as pickle_file:
                self.vectorizer = pickle.load(pickle_file)

            os.remove('trained_nn_model.json')
            os.remove('trained_nn_weights.h5')

    def metrics(self, actual_ratings, predicted_ratings):

        info = {}

        if self.numclasses == 2:

            matrix = confusion_matrix(actual_ratings, predicted_ratings)
            tn, fp, fn, tp = matrix[0][0], matrix[0, 1], matrix[1, 0], matrix[1][1]
            info['tp'], info['tn'], info['fp'], info['fn'] = tp, tn, fp, fn
            info['accuracy'] = (tp + tn) * 1.0 / (tp + tn + fp + fn)
            precision1 = (tp * 1.0) / (tp + fp)
            precision2 = (tn * 1.0) / (tn + fn)
            recall1 = (tp * 1.0) / (tp + fn)
            recall2 = (tn * 1.0) / (tn + fp)
            info['precision1'], info['precision2'], info['recall1'], info['recall2'] = \
                precision1, precision2, recall1, recall2
            info['f1_1'] = 2 * ((precision1 * recall1) / (precision1 + recall1))
            info['f1_2'] = 2 * ((precision2 * recall2) / (precision2 + recall2))

        elif self.numclasses == 3:

            matrix = confusion_matrix(actual_ratings, predicted_ratings)
            tp_pos, tp_neg, tp_neu = matrix[0][0], matrix[1, 1], matrix[2, 2]
            f_ba, f_bc, f_ab = matrix[1, 0], matrix[1, 2], matrix[0, 1]
            f_cb, f_ca, f_ac = matrix[2][1], matrix[2,0], matrix[0, 2]
            info['tp_pos'], info['tp_neg'], info['tp_neut'] = tp_pos, tp_neg, tp_neu
            info['f_ba'], info['f_bc'], info['f_ab'] = f_ba, f_bc, f_ab
            info['f_cb'], info['f_ca'], info['f_ac'] = f_cb, f_ca, f_ac

            info['accuracy'] = ((tp_pos + tp_neg + tp_neu) * 1.0) / \
                               (tp_pos + tp_neg + tp_neu + f_ba + f_bc + f_ab + f_cb + f_ca + f_ac)
            precision1 = (tp_pos * 1.0) / (tp_pos + f_ba + f_ca)
            precision2 = (tp_neg * 1.0) / (tp_neg + f_ab + f_cb)
            precision3 = (tp_neu * 1.0) / (tp_neu + f_bc + f_ac)
            info['precision1'], info['precision2'], info['precision3'] = precision1, precision2, precision3

            recall1 = (tp_pos * 1.0) / (tp_pos + f_ab + f_ac)
            recall2 = (tp_neg * 1.0) / (tp_neg + f_ba + f_bc)
            recall3 = (tp_neu * 1.0) / (tp_neu + f_ca + f_cb)
            info['recall1'], info['recall2'], info['recall3'] = recall1, recall2, recall3

            info['f1_1'] = 2 * ((precision1 * recall1) / (precision1 + recall1))
            info['f1_2'] = 2 * ((precision2 * recall2) / (precision2 + recall2))
            info['f1_3'] = 2 * ((precision3 * recall3) / (precision3 + recall3))

        elif self.numclasses == 5:

            matrix = confusion_matrix(actual_ratings, predicted_ratings)

            tp_one, tp_two, tp_three = matrix[0, 0], matrix[1, 1], matrix[2, 2]
            tp_four, tp_five = matrix[3, 3], matrix[4, 4]
            f_ab, f_ac, f_ad, f_ae = matrix[0, 1], matrix[0, 2], matrix[0, 3], matrix[0, 4]
            f_ba, f_bc, f_bd, f_be = matrix[1, 0], matrix[1, 2], matrix[1, 3], matrix[1, 4]
            f_ca, f_cb, f_cd, f_ce = matrix[2, 0], matrix[2, 1], matrix[2, 3], matrix[2, 4]
            f_da, f_db, f_dc, f_de = matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 4]
            f_ea, f_eb, f_ec, f_ed = matrix[4, 0], matrix[4, 1], matrix[4, 2], matrix[4, 3]
            info['tp_one'], info['tp_two'], info['tp_three'], info['tp_four'], info['tp_five'] = \
                tp_one, tp_two, tp_three, tp_four, tp_five
            info['f_ab'], info['f_ac'], info['f_ad'], info['f_ae'] = f_ab, f_ac, f_ad, f_ae
            info['f_ba'], info['f_bc'], info['f_bd'], info['f_be'] = f_ba, f_bc, f_bd, f_be
            info['f_ca'], info['f_cb'], info['f_cd'], info['f_ce'] = f_ca, f_cb, f_cd, f_ce
            info['f_da'], info['f_db'], info['f_dc'], info['f_de'] = f_da, f_db, f_dc, f_de
            info['f_ea'], info['f_eb'], info['f_ec'], info['f_ed'] = f_ea, f_eb, f_ec, f_ed

            info['accuracy'] = ((tp_one + tp_two + tp_three + tp_four + tp_five) * 1.0) / \
                               (tp_one + tp_two + tp_three + tp_four + tp_five + f_ab + f_ac
                                + f_ad + f_ae + f_ba + f_bc + f_bd + f_be + f_ca + f_cb + f_cd
                                + f_ce + f_da + f_db + f_dc + f_de + f_ea + f_eb + f_ec + f_ed)

            precision1 = (tp_one * 1.0) / (tp_one + f_ba + f_ca + f_da + f_ea)
            precision2 = (tp_two * 1.0) / (tp_two + f_ab + f_cb + f_db + f_eb)
            precision3 = (tp_three * 1.0) / (tp_three + f_ac + f_bc + f_dc + f_ec)
            precision4 = (tp_four * 1.0) / (tp_four + f_ad + f_bd + f_cd + f_ed)
            precision5 = (tp_five * 1.0) / (tp_five + f_ae + f_be + f_ce + f_de)
            info['precision1'], info['precision2'], info['precision3'], info['precision4'], info['precision5'] = \
                precision1, precision2, precision3, precision4, precision5

            recall1 = (tp_one * 1.0) / (tp_one + f_ab + f_ac + f_ad + f_ae)
            recall2 = (tp_two * 1.0) / (tp_two + f_ba + f_bc + f_bd + f_be)
            recall3 = (tp_three * 1.0) / (tp_three + f_ca + f_cb + f_cd + f_ce)
            recall4 = (tp_four * 1.0) / (tp_four + f_da + f_db + f_dc + f_de)
            recall5 = (tp_five * 1.0) / (tp_five + f_ea + f_eb + f_ec + f_ed)
            info['recall1'], info['recall2'], info['recall3'], info['recall4'], info['recall5'] = \
                recall1, recall2, recall3, recall4, recall5

            info['f1_1'] = 2 * ((precision1 * recall1) / (precision1 + recall1))

            if precision2 + recall2 == 0:
                info['f1_2'] = 0

            else:
                info['f1_2'] = 2 * ((precision2 * recall2) / (precision2 + recall2))

            info['f1_3'] = 2 * ((precision3 * recall3) / (precision3 + recall3))
            info['f1_4'] = 2 * ((precision4 * recall4) / (precision4 + recall4))
            info['f1_5'] = 2 * ((precision5 * recall5) / (precision5 + recall5))

        return info

    def optimize_svm(self, data, target):

        cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]

        param_grid = [
            {'C': cs, 'kernel': ['linear']},
            {'C': cs, 'gamma': gammas, 'kernel': ['rbf']}
        ]

        grid = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, cv=2, verbose=2)
        grid.fit(data, target)
        print(grid.best_params_)
        print(grid.best_score_)

    def optimize_rf(self):

        estimators = [10, 100, 500, 1000]
        criterion = ['gini', 'entropy']
        max_depth = [3, None]
        bootstrap = [True, False]
        max_features = ['auto', 'sqrt']

        output = open('examples/rf_results.txt', 'a')

        combos = list(itertools.product(estimators, criterion, max_depth, bootstrap, max_features))

        data, target = self.preprocess('data/common_drugs.csv', count=True)
        skf = StratifiedKFold(n_splits=2)
        indices = list(skf.split(data, target))[0]
        train_indices = indices[0]
        test_indices = indices[1]
        train_data = [data[x] for x in train_indices]
        train_target = [target[x] for x in train_indices]
        test_data = [data[x] for x in test_indices]
        test_target = [target[x] for x in test_indices]

        for i, params in enumerate(combos):

            print('Hyperparameters: {}'.format(params))
            start_time = time.time()
            clf = RandomForestClassifier(n_estimators=params[0],
                                         criterion=params[1],
                                         max_depth=params[2],
                                         bootstrap=params[3],
                                         max_features=params[4])
            clf.fit(train_data, train_target)
            preds = clf.predict(test_data)
            accuracy = accuracy_score(test_target, preds)
            print('Accuracy: {}%'.format(accuracy * 100))
            elapsed = (time.time() - start_time) / 60
            print('Time Elapsed: {0:.2f} min.\n'.format(elapsed))
            output.write('Parameters: {}\nAccuracy: {}%\n'.format(params, accuracy * 100))
            print('Parameter sets trained: {}\nParameter sets remaining: {}'.format(i + 1, len(combos) - (i + 1)))

