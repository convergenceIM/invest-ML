'''
Copyright Convergence Investement Management (2018) All rights reserved.
'''


import pandas as pd
import pandas_datareader.data as web

from sklearn.datasets import make_regression


def get_data(symbol):
    ''' Uses iex as datasource'''
    try:
        df = web.DataReader(symbol, 'iex',start='2014-01-01')
    except:
        raise

    return df

def make_data_regression(begin_date='2014-01-01',end_date='2017-12-31',num_features=5,num_informative=3,noise=1.0,bias=0.1,effective_rank=3,random=42):
    dates = pd.date_range(start='2010-01-01', end='2016-12-31')
    X_vals, y_vals = make_regression(n_samples=len(dates), n_features=num_features, n_informative=num_informative, bias=0.1, noise=1.0,tail_strength =1.0,effective_rank=effective_rank,random_state=random)
    X = pd.DataFrame(X_vals, index=dates)
    y = pd.Series(y_vals, index=dates)
    y.name = 'y_true'
    X.columns = ['f_' + str(i) for i in range(num_features)]
    return X,y

def split_test_train(X,y,pct_test = 0.20):

    X_train = X.iloc[]