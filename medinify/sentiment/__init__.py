"""Configure medinify.scrapers
"""
from medinify.sentiment.review_classifier import ReviewClassifier
from medinify.sentiment.cnn_review_classifier import CNNReviewClassifier, SentimentNetwork
from medinify.sentiment.process import ProcessData

__all__ = (
    'ReviewClassifier',
    'CNNReviewClassifier',
    'SentimentNetwork',
    'ProcessData'
)
