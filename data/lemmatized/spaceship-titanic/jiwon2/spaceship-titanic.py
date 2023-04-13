import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier, cv
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1[_input1.isnull().any(axis=1)]
for col in _input1.columns:
    print(f'{col} columns: {_input1[col].isnull().mean() * 100:.2f}%')
_input1 = _input1.drop(columns=['PassengerId', 'Name'])
ids = _input0['PassengerId']
_input0 = _input0.drop(columns=['PassengerId', 'Name'])
plt.hist(_input1['Age'])
sns.histplot(data=_input1, x='Age', hue='Transported', binwidth=1, kde=True)
_input1['Destination'].value_counts().plot(kind='bar')
_input1['CryoSleep'].value_counts().plot(kind='bar')
_input1['HomePlanet'].value_counts().plot(kind='bar')
column_list = _input1.columns[_input1.isnull().any(axis=0)]
_input1[column_list]
temp = _input1.drop(columns=['Transported'])
total = pd.concat([temp, _input0], axis=0)
total
column_list = column_list.drop('Age')
_input1 = _input1.fillna({col: total[col].mode()[0] for col in column_list}, inplace=False)
_input0 = _input0.fillna({col: total[col].mode()[0] for col in column_list}, inplace=False)
_input1.Age = _input1.Age.fillna(_input1.Age.mean(), inplace=False)
_input0.Age = _input0.Age.fillna(_input1.Age.mean(), inplace=False)
_input1['Deck'] = _input1['Cabin'].apply(lambda x: str(x)[0])
_input1['Num'] = _input1['Cabin'].apply(lambda x: str(x)[2])
_input1['Side'] = _input1['Cabin'].apply(lambda x: str(x)[-1])
_input0['Deck'] = _input0['Cabin'].apply(lambda x: str(x)[0])
_input0['Num'] = _input0['Cabin'].apply(lambda x: str(x)[2])
_input0['Side'] = _input0['Cabin'].apply(lambda x: str(x)[-1])
_input1 = _input1.drop(columns=['Cabin'])
_input0 = _input0.drop(columns=['Cabin'])
_input1.info()
_input1['Num'] = _input1['Num'].astype('int')
_input0['Num'] = _input0['Num'].astype('int')
encoding_columns = _input1.columns[_input1.dtypes == 'object']
encoding_columns
_input1
total = pd.concat([_input1, _input0])
for col in encoding_columns:
    total[col] = pd.factorize(total[col])[0]
_input1 = total[:len(_input1)]
_input0 = total[len(_input1):].drop(columns='Transported')
_input1
corr = _input1.corr()
corr
except_cols = []
remove_cols = []
threshold = 0.4
for col in corr:
    if col in except_cols:
        continue
    except_cols.append(col)
    row = np.abs(corr.loc[col])
    condition1 = row > threshold
    condition2 = ~corr.columns.isin(except_cols)
    temp = row[condition1 & condition2].index
    except_cols = except_cols + list(temp)
    remove_cols = remove_cols + list(temp)
remove_cols
params = {'objective': 'binary:logistic', 'colsample_bytree': 0.7, 'learning_rate': 0.3, 'max_depth': 20, 'alpha': 5, 'eval_metric': 'mlogloss'}
xgb_clf = XGBClassifier(**params)
print(xgb_clf)
y = _input1['Transported'].astype(int)
_input1 = _input1.drop(columns=['Transported'])
dmatrix = xgb.DMatrix(data=_input1, label=y)