import numpy as np
import pandas as pd
import sklearn
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train['dataset'] = 'train'
test['dataset'] = 'test'
df = pd.concat([train, test], axis=0)
df.shape
df['dataset'].value_counts()

def data_desc(df):
    print()
    print('Overall Data Description')
    print('Total number of records', df.shape[0])
    print('Total number of columns/features', df.shape[1])
    print('')
    cols = df.columns
    data_type = []
    for col in df.columns:
        data_type.append(df[col].dtype)
    n_uni = df.nunique()
    n_miss = df.isna().sum()
    names = list(zip(cols, data_type, n_uni, n_miss))
    variable_desc = pd.DataFrame(names, columns=['Name', 'Type', 'Unique levels', 'Missing'])
    print(variable_desc)
data_desc(train)
train['Cabin']
cabin = train['Cabin'].str.split('/', expand=True)
train = pd.concat([train, cabin], axis=1)
train.rename(columns={0: 'Deck', 1: 'Num', 2: 'Port'}, inplace=True)
cabin = test['Cabin'].str.split('/', expand=True)
test = pd.concat([test, cabin], axis=1)
test.rename(columns={0: 'Deck', 1: 'Num', 2: 'Port'}, inplace=True)
train.shape
train.columns
missing_value = train.isnull().sum().sort_values(ascending=False)
missing_perc = (train.isnull().sum() * 100 / train.shape[0]).sort_values(ascending=False)
value = pd.concat([missing_value, missing_perc], axis=1, keys=['Count', '%'])

plt.figure(figsize=(8, 4))
sns.displot(data=train.isna().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', aspect=2)
plt.title('Missing Value Proportion Each Feature')
train_final = train.copy()
data_desc(train_final)
train_final['Num'] = pd.to_numeric(train_final['Num'], errors='coerce')
test['Num'] = pd.to_numeric(test['Num'], errors='coerce')
train_final['VRDeck'].value_counts(dropna=False)
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_final['inhouse_spend'] = train_final['RoomService'] + train_final['Spa'] + train_final['VRDeck']
train_final['out_spend'] = train_final['FoodCourt'] + train_final['ShoppingMall']
test['inhouse_spend'] = test['RoomService'] + test['Spa'] + test['VRDeck']
test['out_spend'] = test['FoodCourt'] + test['ShoppingMall']
train_final['spend_ratio'] = train_final['out_spend'] / train_final['inhouse_spend']
test['spend_ratio'] = test['out_spend'] / test['inhouse_spend']
m = train_final.loc[train_final['spend_ratio'] != np.inf, 'spend_ratio'].max()
train_final['spend_ratio'].replace(np.inf, m, inplace=True)
n = test.loc[test['spend_ratio'] != np.inf, 'spend_ratio'].max()
test['spend_ratio'].replace(np.inf, n, inplace=True)
train_final['spend_ratio'].describe()
train_final['Food_Court_sp'] = train_final['FoodCourt'].apply(lambda x: 1 if x < 300 else 2 if x > 300 else None)
test['Food_Court_sp'] = test['FoodCourt'].apply(lambda x: 1 if x < 300 else 2 if x > 300 else None)
train_final['Food_Court_sp'] = train_final['Food_Court_sp'].astype('category')
test['Food_Court_sp'] = test['Food_Court_sp'].astype('category')
train_final['in_spenders'] = train_final['inhouse_spend'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else None)
test['in_spenders'] = test['inhouse_spend'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else None)
train_final['Age_brac'] = train_final['Age'].apply(lambda x: 1 if x < 1 else 2 if x > 1 else None)
test['Age_brac'] = test['Age'].apply(lambda x: 1 if x < 5 else 2 if x > 5 else None)
train_final['Age_brac'].value_counts(dropna=False)
train_final['spend_brac'] = train_final['inhouse_spend'].apply(lambda x: 1 if x < 185 else 2 if x > 185 else None)
test['spend_brac'] = test['inhouse_spend'].apply(lambda x: 1 if x < 185 else 2 if x > 185 else None)
agg_func_describe = {'inhouse_spend': ['describe']}
train_final.groupby(['Transported']).agg(agg_func_describe).round(2)

def q95(x):
    return x.quantile(0.95)

def q90(x):
    return x.quantile(0.9)
train_final.groupby(['Transported']).agg({'RoomService': ['describe', q90, q95, 'max']})
print('')
train_final.groupby(['Transported']).agg({'Spa': ['describe', q90, q95, 'max']})
train_final.groupby(['Transported']).agg({'VRDeck': ['describe', q90, q95, 'max']})
train_final.groupby(['Transported']).agg({'ShoppingMall': ['describe', q90, q95, 'max']})
train_final.groupby(['Transported']).agg({'FoodCourt': ['describe', q90, q95, 'max']})
train_final.groupby(['Transported']).agg({'inhouse_spend': ['describe', q90, q95, 'max']})
train_final.groupby(['Transported']).agg({'out_spend': ['describe', q90, q95, 'max']})
train_final['RoomService_rev'] = pd.cut(train_final['RoomService'], [0, 470, 4000, 15000])
train_final['Spa_rev'] = pd.cut(train_final['Spa'], [0, 450, 4000, 25000])
train_final['VRDeck_rev'] = pd.cut(train_final['VRDeck'], [0, 400, 5000, 25000])
train_final['FoodCourt_rev'] = pd.cut(train_final['FoodCourt'], [0, 180, 25000, 30000])
train_final['Shop_rev'] = pd.cut(train_final['ShoppingMall'], [0, 100, 620, 25000])
train_final['inspend_rev'] = pd.cut(train_final['inhouse_spend'], [0, 750, 6000, 29000])
train_final['outspend_rev'] = pd.cut(train_final['out_spend'], [0, 60, 1300, 30000])
test['RoomService_rev'] = pd.cut(test['RoomService'], [0, 470, 4000, 15000])
test['Spa_rev'] = pd.cut(test['Spa'], [0, 450, 4000, 25000])
test['VRDeck_rev'] = pd.cut(test['VRDeck'], [0, 400, 5000, 25000])
test['FoodCourt_rev'] = pd.cut(test['FoodCourt'], [0, 180, 25000, 30000])
test['Shop_rev'] = pd.cut(test['ShoppingMall'], [0, 100, 620, 25000])
test['inspend_rev'] = pd.cut(test['inhouse_spend'], [0, 750, 6000, 29000])
test['outspend_rev'] = pd.cut(test['out_spend'], [0, 60, 1300, 30000])
pd.crosstab(train_final['VRDeck_rev'], train_final['Transported'], normalize='columns')
pd.crosstab(train_final['RoomService_rev'], train_final['Transported'], normalize='columns')
pd.crosstab(train_final['spend_brac'], train_final['Transported'])
name = train_final['Name'].str.split(' ', expand=True)
train_final = pd.concat([train_final, name], axis=1)
train_final.rename(columns={0: 'Firstname', 1: 'Lastname'}, inplace=True)
train_final['Firstname'] = train_final['Firstname'].astype('|S')
train_final['Firstname'] = train_final['Firstname'].apply(lambda x: 'None' if x == None else x)
train_final['Lastname'] = train_final['Lastname'].str.replace('[^a-zA-Z]', '')
train_final['Lastname'] = train_final['Lastname'].astype('|S')
train_final['Lastname'] = train_final['Lastname'].apply(lambda x: 'None' if x == None else x)
train_final['Lastname'].isnull().sum()
name = test['Name'].str.split(' ', expand=True)
test = pd.concat([test, name], axis=1)
test.rename(columns={0: 'Firstname', 1: 'Lastname'}, inplace=True)
test['Firstname'] = test['Firstname'].astype('|S')
test['Firstname'] = test['Firstname'].apply(lambda x: 'None' if x == None else x)
test['Lastname'] = test['Lastname'].str.replace('[^a-zA-Z]', '')
test['Lastname'] = test['Lastname'].astype('|S')
test['Lastname'] = test['Lastname'].apply(lambda x: 'None' if x == None else x)
train_final['in_spenders'] = train_final['in_spenders'].astype('category')
test['in_spenders'] = test['in_spenders'].astype('category')
train_final['Age_brac'] = train_final['Age_brac'].astype('category')
test['Age_brac'] = test['Age_brac'].astype('category')
train_final['spend_brac'] = train_final['spend_brac'].astype('category')
test['spend_brac'] = test['spend_brac'].astype('category')
cont_var = []
cat_var = []
for col in train_final.columns:
    if (train_final[col].dtype == 'object') | (train_final[col].dtype == 'category'):
        cat_var.append(col)
    elif (train_final[col].dtype == 'float64') | (train_final[col].dtype == 'int'):
        cont_var.append(col)
train_final[cont_var] = num_imputer.fit_transform(train_final[cont_var])
train_final[cat_var] = cat_imputer.fit_transform(train_final[cat_var])
test[cont_var] = num_imputer.fit_transform(test[cont_var])
test[cat_var] = cat_imputer.fit_transform(test[cat_var])
train_final.hist(bins=12, figsize=(20, 15))

to_remove = ['PassengerId', 'Name', 'Cabin', 'dataset']
cat_var = [i for i in cat_var if i not in to_remove]
train_final['Transported'].value_counts(dropna=False)
train_final['Transported'] = train_final['Transported'].astype('int')
sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})
for var in cont_var:
    sns.barplot(x='Transported', y=var, data=train_final)

train_final.corr()
X = [y for x in [cont_var, cat_var] for y in x]
X
IV = train_final[X]
y = train_final['Transported']
idCol = test.PassengerId.to_numpy()
test.set_index('PassengerId', inplace=True)
from sklearn.preprocessing import LabelEncoder
ohe = OneHotEncoder(handle_unknown='ignore', categories='auto')
label = LabelEncoder()
oe = OrdinalEncoder()
scaler = StandardScaler()
logreg = LogisticRegression(solver='liblinear', multi_class='auto', random_state=1, max_iter=1000)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
ct = make_column_transformer((scaler, ['spend_ratio', 'CryoSleep', 'Age', 'FoodCourt', 'ShoppingMall', 'inhouse_spend', 'Num', 'RoomService', 'Spa', 'VRDeck', 'inhouse_spend', 'out_spend']), (ohe, ['VIP', 'in_spenders', 'Deck', 'Port', 'HomePlanet', 'Destination', 'Age_brac', 'FoodCourt_rev', 'RoomService_rev', 'Spa_rev', 'VRDeck_rev', 'Shop_rev', 'inspend_rev', 'outspend_rev']), remainder='drop')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
pipe_LR = make_pipeline(ct, logreg)