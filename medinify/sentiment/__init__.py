"""Configure medinify.scrapers
"""
from medinify.sentiment.review_classifier import ReviewClassifier
from medinify.sentiment.char_cnn_classifier import CharCnnDataset

__all__ = (
    'ReviewClassifier',
    'CharCnnDataset'
)
