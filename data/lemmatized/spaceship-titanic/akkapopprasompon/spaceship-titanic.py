import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.info()
_input1.describe()

def data_separator(data):
    numeric = data.describe().columns
    data_num = data.loc[:, numeric]
    data_cat = data.drop(numeric, axis=1)
    return (data_num, data_cat)

def num_imputer(data_num):
    data_num = data_num.fillna(data_num.median())
    return data_num

def cat_imputer(data_cat):
    data_cat = data_cat.fillna(data_cat.mode().iloc[0])
    return data_cat
(num, cat) = data_separator(_input1)
target = _input1.iloc[:, -1]
num = num_imputer(num)
cat = cat_imputer(cat)
pd.set_option('display.width', 1000)
print(num.info(), '\n')
print(num.describe().T)
cat.nunique()
pd.set_option('display.max_colwidth', None)
cat.agg({'unique'}).T
cat['Cabin_1'] = cat['Cabin'].apply(lambda x: x.split('/')[0])
cat['Cabin_2'] = cat['Cabin'].apply(lambda x: float(x.split('/')[1]))
cat['Cabin_3'] = cat['Cabin'].apply(lambda x: x.split('/')[2])
cat['PassengerId_2'] = cat['PassengerId'].apply(lambda x: x.split('_')[1])
cat['Transported'] = _input1['Transported']
table = pd.pivot_table(cat[cat.VIP == True], index='Transported', columns=['HomePlanet'], values='PassengerId', aggfunc='count')
cat = cat.drop('Transported', axis=1)
table.plot(kind='bar')
table
' but If you were Mars Citizen you will have a higher chance to not be transported '
cat['Transported'] = _input1['Transported']
table = pd.pivot_table(cat[cat.VIP == True], index='Transported', columns=['Cabin_3'], values='PassengerId', aggfunc='count')
cat = cat.drop('Transported', axis=1)
table.plot(kind='bar')
table
to_drop = ['PassengerId', 'Name', 'Transported', 'Cabin']
cat = cat.loc[:, cat.columns.difference(to_drop)]
cat.agg({'unique'}).T
df = num.join(cat)
(df_num, df_cat) = data_separator(df)
sns.heatmap(df.corr(), cmap='coolwarm', vmax=1)
df_cat_encode = pd.get_dummies(df_cat)
pd.set_option('display.max_colwidth', None)
data_clean = df_num.join(df_cat_encode)
data_clean
X = data_clean
y = target
X = sm.add_constant(X)
model = sm.OLS(y, X.astype(float))