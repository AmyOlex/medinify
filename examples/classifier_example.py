"""
Examples for how to use the Medinify package
"""

from medinify.sentiment import ReviewClassifier, CNNReviewClassifier
import sys
from medinify.sentiment import ProcessData

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

    if sys.argv[1] == 'svm':
        sent = ReviewClassifier('svm')
        process = ProcessData('examples/new_spacy_w2v.model')
        data, target = process.generate_dataset('data/common_drugs.csv')
        sent.optimize_svm(data, target)

    elif sys.argv[1] == 'rf':
        sent = ReviewClassifier('rf')
        data, target = sent.preprocess('data/common_drugs.csv')
        sent.optimize_rf(data, target)

if __name__ == "__main__":
    main()
