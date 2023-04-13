import numpy as np
import pandas as pd
import hyperopt
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input1.info()
_input1.isnull().sum()
_input1 = _input1.fillna(-999, inplace=False)
_input0 = _input0.fillna(-999, inplace=False)
x = _input1.drop('Survived', axis=1)
y = _input1.Survived
x.dtypes
cate_features_index = np.where(x.dtypes != float)[0]
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, train_size=0.85, random_state=1234)
model = CatBoostClassifier(eval_metric='Accuracy', use_best_model=True, random_seed=42)