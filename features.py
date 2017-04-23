import pickle

from nltk import word_tokenize
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, VectorizerMixin
from sklearn.base import BaseEstimator

from utils import load_bw

def get_counts(comments_df, test=False):
    fname = 'count'
    if test:
        fconn = open(fname, 'rb')
        count_vect = pickle.load(fconn)
        fconn.close()
        count_feat = count_vect.transform(comments_df)
    else:
        count_vect = CountVectorizer()
        count_feat = count_vect.fit_transform(comments_df)
        fconn = open(fname, 'wb')
        pickle.dump(count_vect, fconn)
        fconn.close()
    return count_feat


def get_tf(comments_df):
    count_feat = get_counts(comments_df)
    tf_transformer = TfidfTransformer(use_idf=False).fit(count_feat)
    tf_feat = tf_transformer.transform(count_feat)
    return tf_feat
   

def get_tfidf(comments_df, test=False):
    count_feat = get_counts(comments_df, test=test)
    fname = 'tfidf_transformer'
    if test:
        fconn = open(fname, 'rb')
        tfidf_transformer = pickle.load(fconn)
        fconn.close()
        tfidf_feat = tfidf_transformer.transform(count_feat)
        print("Testing shape: {}".format(tfidf_feat.shape))
    else:
        tfidf_transformer = TfidfTransformer()
        tfidf_feat = tfidf_transformer.fit_transform(count_feat)
        fconn = open(fname, 'wb')
        pickle.dump(tfidf_transformer, fconn)
        fconn.close()
        print("Traning shape: {}".format(tfidf_feat.shape))
    return tfidf_feat


class BadWordsCounter(BaseEstimator, VectorizerMixin):

    def __init__(self, badwords_fpath=None):
        print("BadWordsCounter initialized.")
        if badwords_fpath is not None:
            self.badwords_fpath = badwords_fpath
        else:
            self.badwords_fpath = "badwords.txt" 
        self.bw_list = load_bw(self.badwords_fpath)
   
    def get_feature_names(self):
        return np.array([
            'n_words', 'n_chars', 'all_caps', 'max_word_len', 'mean_word_len', 
            'n_bad', 'exclamation_count', 'at_sign_count', 'spaces_count', 
            'all_caps_ratio', 'bad_ratio'
        ])

    def fit(self, X, y=None):
        return self

    def transform(self, comments, y=None):
        print("BadWordsCounter transform starts.")
        df = pd.DataFrame()
        for comment in comments:
            # split comment for later use
            comment_split = comment.split()

            n_words = len(comment_split)
            n_chars = len(comment)
            all_caps = int(np.sum([w.isupper() for w in comment_split]))
            max_word_len = int(np.max([len(w) for w in comment_split]))
            mean_word_len = int(np.mean([len(w) for w in comment_split]))
            n_bad = int(np.sum([comment.lower().count(w) for w in self.bw_list]))
            exclamation_count = comment.count("!")
            at_sign_count = comment.count("@")
            spaces_count = comment.count(" ")
            #all_caps_ratio = np.array(all_caps, dtype=np.float)/np.array(n_words, dtype=np.float)
            all_caps_ratio = float(all_caps)/float(n_words)
            bad_ratio = float(n_bad)/float(n_words)

            row = pd.Series([
                n_words, n_chars, all_caps, max_word_len, mean_word_len, n_bad, 
                exclamation_count, at_sign_count, spaces_count, all_caps_ratio, bad_ratio
            ])
            df = df.append(row, ignore_index=True)
        print("BadWordsCounter transform ends.")
        return df.as_matrix()

