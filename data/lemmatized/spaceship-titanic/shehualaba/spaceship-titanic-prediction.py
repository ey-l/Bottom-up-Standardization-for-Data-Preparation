import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette='colorblind')
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
print(_input1.info())
print('\n')
print(_input1.isnull().sum() / len(_input1))
print('\n')
_input1.shape
_input1['PassengerId'].nunique()
data = _input1.set_index('PassengerId')
data.head()
data['HomePlanet'].value_counts()
print(data['CryoSleep'].nunique())
print(data['CryoSleep'].unique())
data1 = data.copy()
data1['CryoSleep'] = data1['CryoSleep'].astype(bool)
data1['Cabin'].nunique()
cabin_df = data1['Cabin'].str.split('/', expand=True)
cabin_df.nunique()
data2 = data1.copy()
data2[['Deck', 'Side']] = cabin_df.iloc[:, 0:3:2]
data2 = data2.drop('Cabin', axis=1, inplace=False)
data2['Destination'].value_counts()
data2['VIP'].value_counts()
data2['VIP'] = data2['VIP'].astype(bool)
data2['Name'].nunique()
data3 = data2.copy()
data3 = data3.drop('Name', axis=1, inplace=False)
data3.select_dtypes(include='number').describe()
data4 = data3.copy()
data4['Age'] = np.where(data4['Age'] == 0, np.nan, data4['Age'])
data4['Age'].describe()
validated_data = data4.copy()

def validation_func(df):
    data = df.set_index('PassengerId')
    data1 = data.copy()
    data1['CryoSleep'] = data1['CryoSleep'].astype(bool)
    cabin_df = data1['Cabin'].str.split('/', expand=True)
    data2 = data1.copy()
    data2[['Deck', 'Side']] = cabin_df.iloc[:, 0:3:2]
    data2 = data2.drop('Cabin', axis=1, inplace=False)
    data2['VIP'] = data2['VIP'].astype(bool)
    data3 = data2.copy()
    data3 = data3.drop('Name', axis=1, inplace=False)
    data4 = data3.copy()
    data4['Age'] = np.where(data4['Age'] == 0, np.nan, data4['Age'])
    return data4
inputs = validated_data.drop('Transported', axis=1)
targets = validated_data['Transported']

def splitter(df):
    df_cat = df.select_dtypes(exclude='number')
    df_num = df.select_dtypes(include='number')
    return (df_cat, df_num)
(inputs_cat, inputs_num) = splitter(inputs)
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder())])
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_name = inputs_cat.columns
num_name = inputs_num.columns
full_pipeline = ColumnTransformer([('cat_pipeline', cat_pipeline, cat_name), ('num_pipeline', num_pipeline, num_name)])
inputs_cleaned = full_pipeline.fit_transform(inputs)
pca = PCA()
inputs_scaled = pca.fit_transform(inputs_cleaned)
lg_reg = LogisticRegression()