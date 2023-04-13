import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_targets = _input1.pop('Transported')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
data = pd.concat([_input1, _input0])
data['Cabin'] = data['Cabin'].replace(np.NAN, data['Cabin'].mode()[0])
data['Deck'] = data['Cabin'].apply(lambda item: str(item).split('/')[0])
data['Num'] = data['Cabin'].apply(lambda item: str(item).split('/')[1])
data['Side'] = data['Cabin'].apply(lambda item: str(item).split('/')[2])
data.pop('Cabin')
data.pop('PassengerId')
data.pop('Name')
data = pd.get_dummies(data)
_input1 = data.iloc[0:len(_input1)]
_input0 = data.iloc[len(_input1):]
data.head()
models = []
kfold = StratifiedKFold(7, shuffle=True, random_state=2022)
for (index, (train_indices, valid_indices)) in enumerate(kfold.split(_input1, train_targets)):
    x_train = _input1.iloc[train_indices]
    x_val = _input1.iloc[valid_indices]
    y_train = train_targets.iloc[train_indices]
    y_val = train_targets.iloc[valid_indices]
    params = {'iterations': 10000, 'depth': 8, 'early_stopping_rounds': 1000, 'eval_metric': 'Accuracy', 'verbose': 1000}
    model = CatBoostClassifier(**params)