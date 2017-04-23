from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from features import BadWordsCounter
from preprocess import TextPreProcessTransformer


# Pipeline for pulling stats from comment
bw_vec = Pipeline([
        ('stats', BadWordsCounter()),
        ('vect', DictVectorizer())
])

# FeatureUnion to combine bow features and stats features for comment
f_union = FeatureUnion(transformer_list = [
        ('tfidf_vec', TfidfVectorizer(ngram_range=(1, 3), use_idf=False)),
        ('bw_vec', bw_vec)
])


pipeline_lr = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('tfidf_vec', TfidfVectorizer(ngram_range=(1, 3), use_idf=False)),
    ('fs', SelectKBest(chi2, k=20)),
    ('classifier', LogisticRegression())
    # ('clf_svc', SVC(kernel='linear'))
    # ('clf', SGDClassifier(penalty="elasticnet"))
])


pipeline_mnb = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('tfidf_vec', TfidfVectorizer(ngram_range=(1, 3), use_idf=False)),
    # ('fs', SelectKBest(chi2, k=20)),
    ('classifier', MultinomialNB())
    # ('clf_svc', SVC(kernel='linear'))
    # ('clf', SGDClassifier(penalty="elasticnet"))
])


pipeline_svc = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('tfidf_vec', TfidfVectorizer(ngram_range=(1, 3), use_idf=False)),
    # ('fs', SelectKBest(chi2, k=20)),
    # ('fs', SelectKBest(mutual_info_classif, k=1000)),
    # ('classifier', MultinomialNB())
    ('clf_svc', SVC(kernel='linear'))
    # ('clf', SGDClassifier(penalty="elasticnet"))
])

pipeline_svc2 = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union),
    # ('bw', bw_vec),
    # ('tfidf_vec', TfidfVectorizer(ngram_range=(1, 3), use_idf=False)),
    # ('fs', SelectKBest(chi2, k=20)),
    # ('fs', SelectKBest(mutual_info_classif, k=1000)),
    # ('classifier', MultinomialNB())
    ('clf_svc', SVC(kernel='linear'))
    # ('clf', SGDClassifier(penalty="elasticnet"))
])


pipeline_sgdc = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('tfidf_vec', TfidfVectorizer(ngram_range=(1, 3), use_idf=False)),
    ('fs', SelectKBest(chi2, k=20)),
    # ('classifier', MultinomialNB())
    # ('clf_svc', SVC(kernel='linear'))
    ('clf', SGDClassifier())
])
