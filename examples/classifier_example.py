"""
Examples for how to use the Medinify package
"""

from medinify.sentiment import ReviewClassifier, CharCNN, CharCnnDataset, CharCnnNet
import sys
import torch

def main():
    """ Main function.
    """
    # Review sentiment classifier
    # review_classifier = ReviewClassifier('nb') # Try 'nb', 'dt', 'rf', and 'nn'
    # review_classifier.train('citalopram-reviews.csv')
    # review_classifier.save_model()
    # review_classifier.load_model()
    # review_classifier.evaluate_average_accuracy('citalopram-reviews.csv')
    # review_classifier.classify('neutral.txt')

    # data_file = sys.argv[1]

    dataset = CharCnnDataset('data/heart_drugs.csv', 'examples/alphabet.json', 1014, use_medinify_processing=False)

    sent = CharCNN()
    sent.evaluate_k_fold(dataset, 10, 5, 'k-fold-saved.tar')

if __name__ == "__main__":
    main()
