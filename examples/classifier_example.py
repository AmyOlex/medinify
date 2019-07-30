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
        start = int(sys.argv[2])
        sent = ReviewClassifier('svm')
        sent.optimize_svm(start=start)

    elif sys.argv[1] == 'rf':
        n_estimators = int(sys.argv[2])
        criterion = sys.argv[3]
        boostrap = bool(sys.argv[4])
        max_features = sys.argv[5]
        max_depth = None
        if len(sys.argv) == 7:
            max_depth = sys.argv[6]
        sent = ReviewClassifier('rf')
        sent.optimize_rf(n_estimators, criterion, max_depth, boostrap, max_features)

if __name__ == "__main__":
    main()
