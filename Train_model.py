import os
import random
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
import time
import warnings
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')



def model_train(X, y, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std

    spw_base = np.sum(y == 0) / np.sum(y == 1)
    params = {
        'colsample_bylevel': uniform(0, 1),
        'colsample_bynode': uniform(0, 1),
        'colsample_bytree': uniform(0, 1),
        'reg_alpha': loguniform(1e-8, 10),
        'reg_lambda': loguniform(1e-8, 10),
        'min_child_weight': loguniform(1e-2, 10),
        'max_depth': randint(2, 12),
        'scale_pos_weight': [spw_base * 0.5, spw_base, spw_base * 2.0],
        'max_delta_step': randint(0, 10)
    }
    other_params = {
        'learing_rate': 0.05,
        'gamma': 0.1,
        'eval_metric': 'auc',
        'n_jobs': -1,
        'seed': seed,
    }
    est = XGBClassifier(**other_params)
    model = RandomizedSearchCV(
        estimator = est,
        param_distributions = params,
        verbose = 1,
        n_jobs = -1,
        random_state = 42,
        # n_iter = 20,
        # scoring = 'roc_auc,
        # cv = 5
    )

    params = {
        'C': 1,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'penalty': 'l2'
    }
    model = LogisticRegression(**params)

    model.fit(X_norm, y)
    return X_mean, X_std, model



def predict_remove(i, factors, inv, seed=42):
    train_data = factors[inv <= i]
    train0 = train_data[train_data[:, -1] == 0]
    train1 = train_data[train_data[:, -1] == 1]
    train0 = resample(
        train0,
        replace = False,
        n_samples = len(train1),
        random_state = seed
    )
    train_data = np.concatenate((train0, train1))

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_mean, X_std, model = model_train(X_train, y_train)
    X_test = factors[inv == i+1][:, :-1]
    X_test = (X_test - X_mean) / X_std
    y_test = factors[inv == i+1][:, -1]
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print('测试集准确度:', accuracy_score(y_test, y_pred))
    return y_prob


def run_predict(months, factors, n_starts):
    idx = np.argsort(months)
    months = months[idx]
    factors = factors[idx]
    uniq, inv = np.unique(months, return_inverse=True)

    predictions = Parallel(n_jobs=-1)(delayed(predict_remove)(i, factors, inv) for i in range(n_starts, np.max(inv)))
    predictions = np.concatenate([row for row in predictions], axis=0)
    return predictions


