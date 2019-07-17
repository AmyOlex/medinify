
from gensim.models import KeyedVectors
import spacy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords


class ProcessData:

    embeddings = {}
    nlp = None

    def __init__(self, w2v_file):
        self.nlp = spacy.load('en_core_web_sm')
        self.load_vectors(w2v_file)
        self.stops = set(stopwords.words('english'))

    def load_vectors(self, w2v_file):
        w2v = KeyedVectors.load_word2vec_format(w2v_file)
        vectors = dict(zip(list(w2v.vocab.keys()), w2v.vectors))
        self.embeddings = vectors

    def vectorize_comment(self, comment):
        tokens = [token.text for token in self.nlp.tokenizer(comment.lower()) if
                  not token.is_punct and not token.is_space and token not in self.stops]
        vecs = []
        not_found = []
        for token in tokens:
            try:
                vecs.append(self.embeddings[token])
            except KeyError:
                not_found.append(token)
        if len(vecs) == 0:
            average = [0]
        else:
            average = np.mean(vecs, axis=0)
        return average, not_found

    def generate_dataset(self, review_file):
        df = pd.read_csv(review_file)
        target = []
        data = []
        num_discarded = 0
        for review in df.itertuples():
            if type(review.comment) == float or review.rating == 3:
                continue
            rating = 'neg'
            if review.rating in [4, 5]:
                rating = 'pos'
            comment, not_found = self.vectorize_comment(review.comment)
            if len(comment) < 100:
                num_discarded += 1
                continue
            data.append(comment)
            target.append(rating)

        target = LabelEncoder().fit_transform(target)

        print('Processed {} Reviews'.format(len(data)))
        print('{} Reviews discarded because all tokens were out of embedding vocabulary'.format(num_discarded))
        return data, target

    """

    def train(self, review_file):

        df = pd.read_csv(review_file)
        data, target = [], []

        for review in df.itertuples():
            if type(review.comment) == float:
                continue
            if review.rating == 3:
                continue
            comment = ' '.join((re.split('\s+|\t|\n', review.comment.translate(str.maketrans(
                '', '', string.punctuation))))).lower()
            rating = 0
            if review.rating in [4, 5]:
                rating = 1
            data.append(comment)
            target.append(rating)

        vectorizer = CountVectorizer(tokenizer=self.tokenize)
        transformer = TfidfTransformer()
        encoder = LabelEncoder()
        vectors = vectorizer.fit_transform(data)
        frequencied = transformer.fit_transform(vectors)
        labels = encoder.fit_transform(target)

        scores = cross_val_score(RandomForestClassifier(n_estimators=100), vectors, labels, cv=5)
        print(scores)

    def validate_model(self, reviews_file):
        data, target = self.generate_dataset(reviews_file)
        model = RandomForestClassifier(n_estimators=100)
        scores = np.asarray(cross_val_score(model, data, target, cv=5))
        print(scores)
        print(np.mean(scores))
        
    """

    def tokenize(self, text):
        return [token.text for token in self.nlp.tokenizer(text) if token.text not in self.stops]