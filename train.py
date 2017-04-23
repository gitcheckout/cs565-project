from sklearn.metrics import classification_report, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV

from pipelines import (pipeline_abc, pipeline_gnb, pipeline_mnb, pipeline_lr,
                       pipeline_rfc, pipeline_sgdc, pipeline_svc)
from utils import read_data, save_roc_curve


def train_and_report(pipeline, name, data, labels, test_data, test_labels,
                     roc_curve=True):
    print("Train the model {}".format(name))
    pipeline.fit(data, labels)
    print("Model training completed.")

    print("Start predicion..")
    predictions = pipeline.predict(test_data)
    # scores
    accuracy = accuracy_score(test_labels, predictions)
    # auc_score = auc(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    mcc_score = matthews_corrcoef(test_labels, predictions)

    print("Classification Report for {} :".format(name))
    print("----------------------")
    print(classification_report(test_labels, predictions))
    print("----------------------")
    print("F1 score is {}".format(f1))
    print("MCC is {}".format(mcc_score))
    print("Accuracy is {}".format(accuracy))

    if roc_curve:
        # save roc curve
        roc_auc = save_roc_curve(test_labels, predictions)
        print("Area under roc curve: {}".format(roc_auc))
        print("------------------------")


def train(cv=False, roc_curve=True):
    """
    Read data, create test and training sets and run the classifier
    """
    print("Data being read..")
    df = read_data()
    print("Data completely read.")

    # split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
            df['Comment'], df['Insult'], test_size=0.2,
            random_state=11)

    print("Data split completed.")

    pipelines = [
        ("pipeline_abc", pipeline_abc),
        # ("pipeline_gnb", pipeline_gnb),
        # ("pipeline_lr", pipeline_lr),
        ("pipeline_mnb", pipeline_mnb),
        # ("pipeline_rfc", pipeline_rfc),
        ("pipeline_sgdc", pipeline_sgdc),
        ("pipeline_svc", pipeline_svc),
    ]

    for name, pipeline in pipelines:
        # train and predict for all models in pipeline
        if not cv:
            train_and_report(pipeline, name, train_data, train_labels, test_data, test_labels)
        else:
            # if cv is True then do the k-fold cross validation

            # run an exhaustive search of the best parameters on a grid of
            # possible values
            parameters = {
                    'preprocess__stem': (True, False),
                    # 'preprocess__remove_sw': (True, False),
                    # 'preprocess__remove_punct': (True, False),
                    'tfidf_vec__stop_words': (None, 'english'),
                    # 'clf__alpha': (1e-2, 1e-3),
                    # 'clf_svc__kernel': ('linear', 'rbf'),
                    # 'clf_svc__C': (1, 3)
            }

            gs_clf = GridSearchCV(pipeline, parameters, n_jobs=-1)
            gs_clf = gs_clf.fit(train_data, train_labels)

            # display best score
            print(gs_clf.best_score_)
            # display the parameters for which best score was obtained
            print(gs_clf.best_params_)

            # sanity check
            # predict the labels for test data and display score
            print(gs_clf.score(test_data, test_labels))
