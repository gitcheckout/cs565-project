from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from pipelines import pipeline_mnb, pipeline_sgdc, pipeline_svc
from utils import read_data, save_roc_curve


def train(cv=False, roc_curve=True):
    """
    Read data, create test and training sets and run the classifier
    """
    df = read_data()

    train_data, test_data, train_labels, test_labels = train_test_split(
            df['Comment'], df['Insult'], test_size=0.2,
            random_state=11)
    
    pipeline = pipeline_svc
    if not cv:
        pipeline.fit(train_data, train_labels)
        predictions = pipeline.predict(test_data)
        # scores
        # accuracy = accuracy_score(test_labels, predictions)
        # auc_score = auc(test_labels, predictions)
        # f1 = f1_score(test_labels, predictions)
        # mcc_score = matthews_corrcoef(test_labels, predictions)

        print("Classification Report:")
        print("----------------------")
        print(classification_report(test_labels, predictions))
        print("----------------------")

        if roc_curve:
            # save roc curve
            roc_auc = save_roc_curve(test_labels, predictions)
            print("Area under roc curve: {}".format(roc_auc))
    else:
        # run an exhaustive search of the best parameters on a grid of 
        # possible values
        parameters = {
                'preprocess__remove_sw': (True, False),
                'preprocess__remove_punct': (True, False),
                # 'tfidf_vec__use_idf': (True, False),
                # 'tfidf_vec__binary': (True, False),
                # 'clf__alpha': (1e-2, 1e-3),
                # 'clf_svc__kernel': ('linear', 'rbf'),
                # 'clf_svc__C': (1, 3)
        }

        gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, train_labels)

        print(gs_clf.best_score_)
        print(gs_clf.best_params_)
   
        print(gs_clf.score(test_data, test_labels))
