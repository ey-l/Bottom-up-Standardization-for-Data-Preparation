import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Shape of train data: ', _input1.shape)
print('Shape of test data: ', _input0.shape)
X = _input1.drop(columns=['Transported'])
y = _input1['Transported']
values = []
for i in range(len(y)):
    if y[i] == True:
        values.append(1)
    elif y[i] == False:
        values.append(0)
y = pd.DataFrame(data=values, columns=['Transported'])
df = pd.concat([X, _input0], axis=0)
df
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
_input1.info()
df = df.drop(columns=['PassengerId', 'Name'], inplace=False)
cleaner = KNNImputer(n_neighbors=11, weights='distance')
numerical = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df[numerical] = cleaner.fit_transform(df[numerical])
null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
categorical = df[null_columns].select_dtypes(include='object').columns
cleaner = ColumnTransformer([('categorical_transformer', SimpleImputer(strategy='most_frequent'), categorical)])
df[null_columns] = cleaner.fit_transform(df[null_columns])
categorical = df.select_dtypes(include='object').columns
for i in range(0, len(categorical)):
    print(df[categorical[i]].value_counts())
    print('****************************************\n')
cryoSleep_Vip = {False: 0, True: 1}
df['CryoSleep'] = df['CryoSleep'].replace(cryoSleep_Vip)
df['VIP'] = df['VIP'].replace(cryoSleep_Vip)
homeplanet = {'Mars': 0, 'Earth': 1, 'Europa': 2}
df['HomePlanet'] = df['HomePlanet'].replace(homeplanet)
destination = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}
df['Destination'] = df['Destination'].replace(destination)
encoder = preprocessing.LabelEncoder()
df['Cabin'] = encoder.fit_transform(df['Cabin'].astype(str))
df
X = df.iloc[:8693, :]
_input0 = df.iloc[8693:, :]
new_train = pd.concat([X, y], axis=1)
new_train
new_train['Transported'].value_counts()
sns.countplot(x=new_train['Transported'])
df_minority_0 = new_train[new_train['Transported'] == 0]
df_majority_1 = new_train[new_train['Transported'] == 1]
df_minority_upsampled = resample(df_minority_0, replace=True, n_samples=4378, random_state=42)
df_upsampled = pd.concat([df_minority_upsampled, df_majority_1])
sns.countplot(x=df_upsampled['Transported'])
df_upsampled['Transported'].value_counts()
plt.figure(figsize=(15, 15))
cor = df_upsampled.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, fmt='.2f')
X = df_upsampled.drop(columns=['Transported'])
y = df_upsampled['Transported']
from sklearn.feature_selection import SelectKBest, chi2
FeatureSelection = SelectKBest(score_func=chi2, k=7)
X = FeatureSelection.fit_transform(X, y)
print('X Shape is ', X.shape)
print('Selected Features are : ', FeatureSelection.get_support())
cols = ['CryoSleep', 'Cabin', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(X_train, X_val_test, y_train, y_val_test) = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=150)
(X_val, X_test, y_val, y_test) = train_test_split(X_val_test, y_val_test, train_size=0.5, shuffle=True, random_state=150)
RandomForestClassifierModel = RandomForestClassifier(criterion='entropy', max_depth=13, n_estimators=150, random_state=150)