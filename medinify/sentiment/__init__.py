"""Configure medinify.scrapers
"""
from medinify.sentiment.review_classifier import ReviewClassifier
from medinify.sentiment.char_cnn_dataset import CharCnnDataset
from medinify.sentiment.char_cnn_model import CharCnnNet
from medinify.sentiment.char_cnn import CharCNN

__all__ = (
    'ReviewClassifier',
    'CharCnnDataset',
    'CharCnnNet',
    'CharCNN'
)
