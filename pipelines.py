from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

from features import BadWordsCounter
from preprocess import TextPreProcessTransformer
from utils import DenseTransformer


# Pipeline for pulling stats from comment
bw_vec = Pipeline([
        ('stats', BadWordsCounter()),
        ('vect', DictVectorizer())
])

tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), use_idf=False)

# some classifiers require dense data
bw_vec_dense = Pipeline([
        ('stats', BadWordsCounter()),
        ('vect', DictVectorizer()),
        ('dense', DenseTransformer())
])

tfidf_vec_dense = Pipeline([
    ('tfidf_vec', tfidf_vec),
    ('dense', DenseTransformer())
])


# FeatureUnion to combine bow features and stats features for comment
f_union = FeatureUnion(transformer_list=[
        ('tfidf_vec', tfidf_vec),
        ('bw_vec', bw_vec)
])

f_union_dense = FeatureUnion(transformer_list=[
        ('tfidf_vec', tfidf_vec_dense),
        ('bw_vec', bw_vec_dense)
])


pipeline_abc = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union),
    # ('fs', SelectKBest(chi2, k=20)),
    ('clf', AdaBoostClassifier())
])


pipeline_lr = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union),
    # ('fs', SelectKBest(chi2, k=20)),
    ('classifier', LogisticRegression(C=3))
])


pipeline_mnb = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union),
    # ('fs', SelectKBest(chi2, k=20)),
    ('classifier', MultinomialNB())
])


pipeline_gnb = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union_dense),
    ('fs', SelectKBest(chi2, k=1000)),
    ('classifier', GaussianNB())
])


pipeline_svc = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union),
    # ('fs', SelectKBest(chi2, k=20)),
    # ('fs', SelectKBest(mutual_info_classif, k=1000)),
    ('clf_svc', SVC(kernel='linear'))
])


pipeline_sgdc = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union),
    # ('fs', SelectKBest(chi2, k=20)),
    ('clf', SGDClassifier())
])


pipeline_rfc = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union_dense),
    ('fs', SelectKBest(chi2, k=1000)),
    ('clf', RandomForestClassifier(min_samples_split=5))
])


pipeline_vc = Pipeline([
    ('preprocess', TextPreProcessTransformer(stem=True)),
    ('features', f_union),
    ('clf', VotingClassifier(
      estimators=[
          ('lr', LogisticRegression(C=3 )),
          ('svc', SVC(C=0.3, kernel="linear", probability=True))
      ], n_jobs=-1, voting="soft", weights=[0.4, 0.6]
    ))
])
