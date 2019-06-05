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

    data_file = sys.argv[1]

    dataset = CharCnnDataset(data_file, 'examples/alphabet.json', 1014, use_medinify_processing=False)

    sent = CharCNN()
    loader = CharCNN.get_data_loader(dataset, 25)

    network = CharCnnNet()

    with open('examples/kfold_file_example.tar', 'rb') as f:
        info = torch.load(f)
        print(info)

if __name__ == "__main__":
    main()
