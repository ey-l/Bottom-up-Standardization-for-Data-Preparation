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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1['dataset'] = 'train'
_input0['dataset'] = 'test'
df = pd.concat([_input1, _input0], axis=0)
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
data_desc(_input1)
_input1['Cabin']
cabin = _input1['Cabin'].str.split('/', expand=True)
_input1 = pd.concat([_input1, cabin], axis=1)
_input1 = _input1.rename(columns={0: 'Deck', 1: 'Num', 2: 'Port'}, inplace=False)
cabin = _input0['Cabin'].str.split('/', expand=True)
_input0 = pd.concat([_input0, cabin], axis=1)
_input0 = _input0.rename(columns={0: 'Deck', 1: 'Num', 2: 'Port'}, inplace=False)
_input1.shape
_input1.columns
missing_value = _input1.isnull().sum().sort_values(ascending=False)
missing_perc = (_input1.isnull().sum() * 100 / _input1.shape[0]).sort_values(ascending=False)
value = pd.concat([missing_value, missing_perc], axis=1, keys=['Count', '%'])
plt.figure(figsize=(8, 4))
sns.displot(data=_input1.isna().melt(value_name='missing'), y='variable', hue='missing', multiple='fill', aspect=2)
plt.title('Missing Value Proportion Each Feature')
train_final = _input1.copy()
data_desc(train_final)
train_final['Num'] = pd.to_numeric(train_final['Num'], errors='coerce')
_input0['Num'] = pd.to_numeric(_input0['Num'], errors='coerce')
train_final['VRDeck'].value_counts(dropna=False)
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_final['inhouse_spend'] = train_final['RoomService'] + train_final['Spa'] + train_final['VRDeck']
train_final['out_spend'] = train_final['FoodCourt'] + train_final['ShoppingMall']
_input0['inhouse_spend'] = _input0['RoomService'] + _input0['Spa'] + _input0['VRDeck']
_input0['out_spend'] = _input0['FoodCourt'] + _input0['ShoppingMall']
train_final['spend_ratio'] = train_final['out_spend'] / train_final['inhouse_spend']
_input0['spend_ratio'] = _input0['out_spend'] / _input0['inhouse_spend']
m = train_final.loc[train_final['spend_ratio'] != np.inf, 'spend_ratio'].max()
train_final['spend_ratio'] = train_final['spend_ratio'].replace(np.inf, m, inplace=False)
n = _input0.loc[_input0['spend_ratio'] != np.inf, 'spend_ratio'].max()
_input0['spend_ratio'] = _input0['spend_ratio'].replace(np.inf, n, inplace=False)
train_final['spend_ratio'].describe()
train_final['Food_Court_sp'] = train_final['FoodCourt'].apply(lambda x: 1 if x < 300 else 2 if x > 300 else None)
_input0['Food_Court_sp'] = _input0['FoodCourt'].apply(lambda x: 1 if x < 300 else 2 if x > 300 else None)
train_final['Food_Court_sp'] = train_final['Food_Court_sp'].astype('category')
_input0['Food_Court_sp'] = _input0['Food_Court_sp'].astype('category')
train_final['in_spenders'] = train_final['inhouse_spend'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else None)
_input0['in_spenders'] = _input0['inhouse_spend'].apply(lambda x: 1 if x > 0 else 0 if x == 0 else None)
train_final['Age_brac'] = train_final['Age'].apply(lambda x: 1 if x < 1 else 2 if x > 1 else None)
_input0['Age_brac'] = _input0['Age'].apply(lambda x: 1 if x < 5 else 2 if x > 5 else None)
train_final['Age_brac'].value_counts(dropna=False)
train_final['spend_brac'] = train_final['inhouse_spend'].apply(lambda x: 1 if x < 185 else 2 if x > 185 else None)
_input0['spend_brac'] = _input0['inhouse_spend'].apply(lambda x: 1 if x < 185 else 2 if x > 185 else None)
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
_input0['RoomService_rev'] = pd.cut(_input0['RoomService'], [0, 470, 4000, 15000])
_input0['Spa_rev'] = pd.cut(_input0['Spa'], [0, 450, 4000, 25000])
_input0['VRDeck_rev'] = pd.cut(_input0['VRDeck'], [0, 400, 5000, 25000])
_input0['FoodCourt_rev'] = pd.cut(_input0['FoodCourt'], [0, 180, 25000, 30000])
_input0['Shop_rev'] = pd.cut(_input0['ShoppingMall'], [0, 100, 620, 25000])
_input0['inspend_rev'] = pd.cut(_input0['inhouse_spend'], [0, 750, 6000, 29000])
_input0['outspend_rev'] = pd.cut(_input0['out_spend'], [0, 60, 1300, 30000])
pd.crosstab(train_final['VRDeck_rev'], train_final['Transported'], normalize='columns')
pd.crosstab(train_final['RoomService_rev'], train_final['Transported'], normalize='columns')
pd.crosstab(train_final['spend_brac'], train_final['Transported'])
name = train_final['Name'].str.split(' ', expand=True)
train_final = pd.concat([train_final, name], axis=1)
train_final = train_final.rename(columns={0: 'Firstname', 1: 'Lastname'}, inplace=False)
train_final['Firstname'] = train_final['Firstname'].astype('|S')
train_final['Firstname'] = train_final['Firstname'].apply(lambda x: 'None' if x == None else x)
train_final['Lastname'] = train_final['Lastname'].str.replace('[^a-zA-Z]', '')
train_final['Lastname'] = train_final['Lastname'].astype('|S')
train_final['Lastname'] = train_final['Lastname'].apply(lambda x: 'None' if x == None else x)
train_final['Lastname'].isnull().sum()
name = _input0['Name'].str.split(' ', expand=True)
_input0 = pd.concat([_input0, name], axis=1)
_input0 = _input0.rename(columns={0: 'Firstname', 1: 'Lastname'}, inplace=False)
_input0['Firstname'] = _input0['Firstname'].astype('|S')
_input0['Firstname'] = _input0['Firstname'].apply(lambda x: 'None' if x == None else x)
_input0['Lastname'] = _input0['Lastname'].str.replace('[^a-zA-Z]', '')
_input0['Lastname'] = _input0['Lastname'].astype('|S')
_input0['Lastname'] = _input0['Lastname'].apply(lambda x: 'None' if x == None else x)
train_final['in_spenders'] = train_final['in_spenders'].astype('category')
_input0['in_spenders'] = _input0['in_spenders'].astype('category')
train_final['Age_brac'] = train_final['Age_brac'].astype('category')
_input0['Age_brac'] = _input0['Age_brac'].astype('category')
train_final['spend_brac'] = train_final['spend_brac'].astype('category')
_input0['spend_brac'] = _input0['spend_brac'].astype('category')
cont_var = []
cat_var = []
for col in train_final.columns:
    if (train_final[col].dtype == 'object') | (train_final[col].dtype == 'category'):
        cat_var.append(col)
    elif (train_final[col].dtype == 'float64') | (train_final[col].dtype == 'int'):
        cont_var.append(col)
train_final[cont_var] = num_imputer.fit_transform(train_final[cont_var])
train_final[cat_var] = cat_imputer.fit_transform(train_final[cat_var])
_input0[cont_var] = num_imputer.fit_transform(_input0[cont_var])
_input0[cat_var] = cat_imputer.fit_transform(_input0[cat_var])
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
idCol = _input0.PassengerId.to_numpy()
_input0 = _input0.set_index('PassengerId', inplace=False)
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