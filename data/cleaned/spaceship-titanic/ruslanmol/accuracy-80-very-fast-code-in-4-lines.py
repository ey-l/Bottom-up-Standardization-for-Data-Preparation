from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
combine = [df_train, df_test]
from collections import Counter
for dataset in combine:
    dataset['ID1'] = dataset.PassengerId.str.split('_', n=1, expand=True)[0]
    dataset['ID2'] = dataset.PassengerId.str.split('_', n=1, expand=True)[1]
    dataset['ID1'] = dataset['ID1'].astype(int)
    dataset['ID2'] = dataset['ID2'].astype(int)
    dataset['Name1'] = dataset.Name.str.split(' ', n=1, expand=True)[0]
    CountName1 = Counter(dataset['Name1'])
    dataset['Name1Count'] = dataset['Name1'].map(CountName1)
    dataset['NameLen1'] = dataset['Name1'].str.count('\\S')
    dataset['Name2'] = dataset.Name.str.split(' ', n=1, expand=True)[1]
    CountName2 = Counter(dataset['Name2'])
    dataset['Name2Count'] = dataset['Name2'].map(CountName2)
    dataset['VIP'] = dataset['VIP'].replace(np.nan, -1).astype(int)
    dataset['VIP'] = dataset['VIP'].replace(-1, np.nan)
    dataset['CryoSleep'] = dataset['CryoSleep'].replace(np.nan, -1).astype(int)
    dataset['CryoSleep'] = dataset['CryoSleep'].replace(-1, np.nan)
    dataset['Cabin1'] = dataset.Cabin.str.split('/', n=2, expand=True)[0]
    dataset['Cabin2'] = dataset.Cabin.str.split('/', n=2, expand=True)[1]
    dataset['Cabin2'] = dataset['Cabin2'].replace(np.nan, -1).astype(int)
    dataset['Cabin2'] = dataset['Cabin2'].replace(-1, np.nan)
    dataset['Cabin3'] = dataset.Cabin.str.split('/', n=2, expand=True)[2]
    dataset.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)

def missing_values_table(dataframe, na_name=True):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum()
    miss_dtypes = dataframe[na_columns].dtypes
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio, miss_dtypes], axis=1, keys=['Missing_Values (#)', 'Ratio (%)', 'Type'])
    print(missing_df.sort_values('Ratio (%)', ascending=False))
    print('************* Number of Missing Values *************')
    print(dataframe.isnull().sum().sum())
    if na_name:
        return na_columns
na_columns = missing_values_table(df_train, na_name=True)
Transformer = SimpleImputer(strategy='most_frequent')
for dataset in combine:
    df_filled = Transformer.fit_transform(dataset[na_columns])
    dataset[na_columns] = pd.DataFrame(df_filled, columns=dataset[na_columns].columns)
for dataset in combine:
    for col in df_test.columns:
        try:
            dataset[col] = dataset[col].astype('float64')
        except:
            print('object column:', col)
X = df_train.drop('Transported', axis=1)
y = df_train['Transported'].astype(int)
print('Please wait 1 minute...simulating...')