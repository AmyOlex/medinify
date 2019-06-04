"""
Examples for how to use the Medinify package
"""

from medinify.sentiment import ReviewClassifier, CharCNN, CharCnnDataset, CharCnnNet
import sys

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

    data_file = sys.argv[1]

    dataset = CharCnnDataset(data_file, 'examples/alphabet.json', 1014, use_medinify_processing=False)
    loader = CharCNN.get_data_loader(dataset, 25)

    network = CharCnnNet()
    sent = CharCNN()

    sent.train(network, train_loader=loader, n_epochs=10)

if __name__ == "__main__":
    main()
