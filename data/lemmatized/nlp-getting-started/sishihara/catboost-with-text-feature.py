import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

def label_encoding(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict):
    """
    col_definition: encode_col
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    for f in col_definition['encode_col']:
        try:
            lbl = preprocessing.LabelEncoder()
            train[f] = lbl.fit_transform(list(train[f].values))
        except:
            print(f)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return (train, test)
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input1.head()
target_col = 'target'
text_cols = ['text']
categorical_cols = ['keyword', 'location']
(_input1, _input0) = label_encoding(_input1, _input0, col_definition={'encode_col': categorical_cols})
X_train = _input1[text_cols + categorical_cols]
y_train = _input1[target_col].values
X_test = _input0[text_cols + categorical_cols]
y_preds = []
models = []
oof_train = np.zeros((len(X_train),))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
catboost_params = {'iterations': 1000, 'learning_rate': 0.1, 'eval_metric': 'Logloss', 'task_type': 'GPU', 'early_stopping_rounds': 10, 'use_best_model': True, 'verbose': 100}
for (fold_id, (train_index, valid_index)) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]
    train_pool = Pool(X_tr, y_tr, cat_features=categorical_cols, text_features=text_cols, feature_names=list(X_tr))
    valid_pool = Pool(X_val, y_val, cat_features=categorical_cols, text_features=text_cols, feature_names=list(X_tr))
    model = CatBoostClassifier(**catboost_params)