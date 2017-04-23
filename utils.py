import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

from constants import *


def read_data(drop_date=True, csv_fpath=TRAINING_CSV):
    df = pd.read_csv(csv_fpath)
    if drop_date and 'Date' in df.columns.values:
        df = df.drop(labels=['Date'], axis=1)
    return df


def load_bw(fpath):
    fconn = open(fpath, 'r')
    bw_list = []
    for i, line in enumerate(fconn):
        if i != 0:
            insult = (line.strip().split(":")[0]).strip('"')
            bw_list.append(insult)
    #print(len(bw_list))
    #print(bw_list)
    return bw_list


def save_roc_curve(actual, predictions, fname="roc.png"):
    fp_rate, tp_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(fp_rate, tp_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fp_rate, tp_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(fname, bbox_inches='tight')
    # plt.show()
    return roc_auc
