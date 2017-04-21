import pandas as pd

from constants import *


def read_data(drop_date=True, csv_fpath=TRAINING_CSV):
    df= pd.read_csv(csv_fpath)
    if 'Date' in df.columns.values:
        df = df.drop(labels=['Date'], axis=1)
    return df

