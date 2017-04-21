import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocess import TextPreProcessTransformer


pipeline = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    # TfidfVectorizer is equivalent to CountVectorizer followed 
    # by TfidfTransformer.
    ('tfidf_vec', TfidfVectorizer(ngram_range=(1, 3), use_idf=False)),
    #('classifier', MultinomialNB())
    #('classifier', SVC(kernel='linear', probability=True))
    ('clf', SGDClassifier())
])

