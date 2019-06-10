"""Configure medinify.scrapers
"""
from medinify.sentiment.review_classifier import ReviewClassifier
from medinify.sentiment.cnn_review_classifier import CNNReviewClassifier
from medinify.sentiment.cnn_network import SentimentNetwork
from medinify.sentiment.cnn_dataset import SentimentDataset

__all__ = (
    'ReviewClassifier',
    'CNNReviewClassifier',
    'SentimentNetwork',
    'SentimentDataset'
)
