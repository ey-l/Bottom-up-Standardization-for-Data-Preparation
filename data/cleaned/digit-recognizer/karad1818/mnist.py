import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn import model_selection
from sklearn import tree
from sklearn import metrics

def create_fold(data):
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=6)
    for (fold, (t_, v_)) in enumerate(kf.split(X=data, y=data.label.values)):
        data.loc[v_, 'kfold'] = fold
    return data

def run(data, fold, depth):
    df_train = data[data.kfold != fold].reset_index(drop=True)
    df_val = data[data.kfold == fold].reset_index(drop=True)
    X_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values
    X_val = df_val.drop('label', axis=1).values
    y_val = df_val.label.values
    model = tree.DecisionTreeClassifier(max_depth=depth)