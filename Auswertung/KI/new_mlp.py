import time
import pandas as pd
import pickle as pk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score, confusion_matrix

start_time = time.time()

kfold = 10
filename = 'csv/dataset_488.csv'


def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop.sort_values(ascending=False))
    return au_corr[0:n]


def corrank(X):
    import itertools
    df = pd.DataFrame([[(i, j),
                        X.corr().loc[i, j]] for i, j in list(itertools.combinations(X.corr(), 2))],
                      columns=['pairs', 'corr'])
    print(df.sortvalues(by='corr', ascending=False))
    print()