import pandas as pd
pd.set_option('display.max_columns', None)
data = pd.read_csv('data/input/spaceship-titanic/train.csv')
data_ind_var = data.iloc[:, :-1]
data_ind_var['train'] = 1
data_ind_var
counts = data.iloc[:, :-1].nunique()
counts
data_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
data_test['train'] = 0
data_test
data_combined = pd.concat([data_ind_var, data_test])
data_combined
data_combined['group'] = data_combined.PassengerId.str.split('_').str[0]
data_combined['group_count'] = data_combined.PassengerId.str.split('_').str[1]
data_combined['deck'] = data_combined.Cabin.str.split('/').str[0]
data_combined['cabin_number'] = data_combined.Cabin.str.split('/').str[1]
data_combined['cabin_side'] = data_combined.Cabin.str.split('/').str[2]
data_combined
data_combined.drop(['PassengerId', 'Cabin', 'Name'], axis=1, inplace=True)
data_combined
convert_dict = {'group': int, 'group_count': int}
data_combined.astype(convert_dict)
columns_list = list(data_combined.columns)
data_combined = data_combined[columns_list[0:10] + columns_list[11:] + [columns_list[10]]]
data_combined
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
A = make_column_transformer((OneHotEncoder(categories='auto', drop='first'), ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'deck', 'cabin_side']), remainder='passthrough')
X = A.fit_transform(data_combined)
X
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')