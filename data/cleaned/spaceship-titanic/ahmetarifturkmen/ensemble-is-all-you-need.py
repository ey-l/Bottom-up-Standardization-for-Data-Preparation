import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
PATH = 'data/input/spaceship-titanic'
train = pd.read_csv(f'{PATH}/train.csv')
test = pd.read_csv(f'{PATH}/test.csv')
train.head()
train.info()
train['Transported'].replace(False, 0, inplace=True)
train['Transported'].replace(True, 1, inplace=True)
train.Transported = train.Transported.astype(int)
train['VIP'].replace(False, 0, inplace=True)
train['VIP'].replace(True, 1, inplace=True)
test['VIP'].replace(False, 0, inplace=True)
test['VIP'].replace(True, 1, inplace=True)
test.VIP = test.VIP.astype('Int8')
train.VIP = train.VIP.astype('Int8')
train.CryoSleep.replace(False, 0, inplace=True)
test.CryoSleep.replace(False, 0, inplace=True)
train.CryoSleep.replace(True, 1, inplace=True)
test.CryoSleep.replace(True, 1, inplace=True)
test.CryoSleep = test.CryoSleep.astype('Int8')
train.CryoSleep = train.CryoSleep.astype('Int8')
train[['deck', 'num', 'side']] = train['Cabin'].str.split('/', expand=True)
test[['deck', 'num', 'side']] = test['Cabin'].str.split('/', expand=True)
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train['SumSpends'] = train[col_to_sum].sum(axis=1)
test['SumSpends'] = test[col_to_sum].sum(axis=1)
train['MaxSpends'] = train[col_to_sum].max(axis=1)
test['MaxSpends'] = test[col_to_sum].max(axis=1)
train['log_spend'] = np.log(train.SumSpends + 1)
test['log_spend'] = np.log(test.SumSpends + 1)
null_cols = train.isnull().sum().sort_values(ascending=False)
null_cols = list(null_cols[null_cols > 1].index)
object_cols = [col for col in train.columns if train[col].dtype == 'object' or train[col].dtype == 'category']
from sklearn.preprocessing import OrdinalEncoder
oc = OrdinalEncoder()
df_for_encode = pd.concat([train, test])
df_for_encode[object_cols] = df_for_encode[object_cols].astype('category')
df_for_encode[object_cols] = oc.fit_transform(df_for_encode[object_cols])
del train, test
train = df_for_encode.iloc[:8693, :]
test = df_for_encode.iloc[8693:, :]
del df_for_encode
test.drop('Transported', inplace=True, axis=1)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('imp', SimpleImputer(strategy='mean'), null_cols)])
train[null_cols] = ct.fit_transform(train[null_cols])
test[null_cols] = ct.transform(test[null_cols])
train.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
y_train = train['Transported']
X_train = train.drop('Transported', axis=1)
X_test = test
if X_train.shape[1] == X_test.shape[1]:
    print('Shapes are equal. We are ready to train models.')
else:
    print('There is something wrong in preprocessing steps.')
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=34)
fold_accuracy = []
for (tr_idx, val_idx) in cv.split(X_train, y_train):
    (X_tr, y_tr, X_val, y_val) = (X_train.iloc[tr_idx], y_train.iloc[tr_idx], X_train.iloc[val_idx], y_train.iloc[val_idx])
    model = CatBoostClassifier(eval_metric='Accuracy', verbose=0, rsm=0.82, iterations=700)