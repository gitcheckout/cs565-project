import os

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

from features import get_tfidf
from pipelines import pipeline
from preprocess import TextPreProcessTransformer
from utils import read_data

def train():
    df = read_data()

    train_data, test_data, train_labels, test_labels = train_test_split(
            df['Comment'], df['Insult'], test_size=0.2,
            random_state=11)
    #pipeline.fit(train_data, train_labels)
    #predictions = pipeline.predict(test_data)
    
    # run an exhaustive search of the best parameters on a grid of 
    # possible values
    parameters = {
            #'preprocess__stem': (True, False),
            #'tfidf_vec__use_idf': (True, False),
            'tfidf_vec__binary': (True, False),
            'clf__alpha': (1e-2, 1e-3),
    }
    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_data, train_labels)
    #predictions = gs_clf.predict(test_labels)

    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    #score = f1_score(test_labels, predictions)
    #print(score)
   
    print(gs_clf.score(test_data, test_labels))

