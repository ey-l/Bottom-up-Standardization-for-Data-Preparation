import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier, cv
X_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
X_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
X_train[X_train.isnull().any(axis=1)]
for col in X_train.columns:
    print(f'{col} columns: {X_train[col].isnull().mean() * 100:.2f}%')
X_train = X_train.drop(columns=['PassengerId', 'Name'])
ids = X_test['PassengerId']
X_test = X_test.drop(columns=['PassengerId', 'Name'])
plt.hist(X_train['Age'])

sns.histplot(data=X_train, x='Age', hue='Transported', binwidth=1, kde=True)
X_train['Destination'].value_counts().plot(kind='bar')

X_train['CryoSleep'].value_counts().plot(kind='bar')

X_train['HomePlanet'].value_counts().plot(kind='bar')

column_list = X_train.columns[X_train.isnull().any(axis=0)]
X_train[column_list]
temp = X_train.drop(columns=['Transported'])
total = pd.concat([temp, X_test], axis=0)
total
column_list = column_list.drop('Age')
X_train.fillna({col: total[col].mode()[0] for col in column_list}, inplace=True)
X_test.fillna({col: total[col].mode()[0] for col in column_list}, inplace=True)
X_train.Age.fillna(X_train.Age.mean(), inplace=True)
X_test.Age.fillna(X_train.Age.mean(), inplace=True)
X_train['Deck'] = X_train['Cabin'].apply(lambda x: str(x)[0])
X_train['Num'] = X_train['Cabin'].apply(lambda x: str(x)[2])
X_train['Side'] = X_train['Cabin'].apply(lambda x: str(x)[-1])
X_test['Deck'] = X_test['Cabin'].apply(lambda x: str(x)[0])
X_test['Num'] = X_test['Cabin'].apply(lambda x: str(x)[2])
X_test['Side'] = X_test['Cabin'].apply(lambda x: str(x)[-1])
X_train = X_train.drop(columns=['Cabin'])
X_test = X_test.drop(columns=['Cabin'])
X_train.info()
X_train['Num'] = X_train['Num'].astype('int')
X_test['Num'] = X_test['Num'].astype('int')
encoding_columns = X_train.columns[X_train.dtypes == 'object']
encoding_columns
X_train
total = pd.concat([X_train, X_test])
for col in encoding_columns:
    total[col] = pd.factorize(total[col])[0]
X_train = total[:len(X_train)]
X_test = total[len(X_train):].drop(columns='Transported')
X_train
corr = X_train.corr()
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
y = X_train['Transported'].astype(int)
X_train = X_train.drop(columns=['Transported'])
dmatrix = xgb.DMatrix(data=X_train, label=y)