import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.shape
_input1.columns.tolist()
_input1.dtypes
_input1['TotalExp'] = _input1['RoomService'] + _input1['FoodCourt'] + _input1['ShoppingMall'] + _input1['Spa'] + _input1['VRDeck']
_input0['TotalExp'] = _input0['RoomService'] + _input0['FoodCourt'] + _input0['ShoppingMall'] + _input0['Spa'] + _input0['VRDeck']
_input1.head()
_input1[['CabinDeck', 'CabinNum', 'CabinSide']] = _input1['Cabin'].str.split('/', expand=True)
_input0[['CabinDeck', 'CabinNum', 'CabinSide']] = _input0['Cabin'].str.split('/', expand=True)
_input1.head()
_input1[['PassengerGrp', 'PassengerNum']] = _input1['PassengerId'].str.split('_', expand=True).astype(int)
_input0[['PassengerGrp', 'PassengerNum']] = _input0['PassengerId'].str.split('_', expand=True).astype(int)
_input1.head()
for col in _input1.columns:
    print(f'{col} ---------> {_input1[col].nunique()}')
for col in _input0.columns:
    print(f'{col} ---------> {_input0[col].nunique()}')
num_features = _input1.select_dtypes(exclude=['object', 'bool']).columns.tolist()
_input1[num_features].head()
cat_features = _input1.select_dtypes(include=['object']).columns.tolist()
_input1[cat_features].head()
_input1.describe()
transported_df = _input1.copy()
transported_df['Transported'] = transported_df['Transported'].map({True: 'Transported', False: 'Not Transported'})
feat_cat = ['HomePlanet', 'Destination']
(fig, axes) = plt.subplots(2, 1, figsize=(15, 10))
fig.subplots_adjust(hspace=0.4)
i = 0
for triaxis in axes:
    sns.countplot(data=transported_df, x='Transported', hue=transported_df[feat_cat[i]], palette='flare', ax=triaxis)
    i = i + 1
(fig, axes) = plt.subplots(2, 2, figsize=(18, 15))
fig.subplots_adjust(hspace=0.4)
feat_cat = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']
i = 0
for triaxis in axes:
    for axis in triaxis:
        sns.countplot(data=transported_df, x=_input1[feat_cat[i]], hue='CryoSleep', palette='flare', ax=axis)
        i = i + 1
(fig, axes) = plt.subplots(len(_input1[num_features].columns) // 4, 4, figsize=(18, 10))
i = 0
fig.subplots_adjust(hspace=0.5, wspace=0.5)
for triaxis in axes:
    for axis in triaxis:
        sns.kdeplot(data=_input1[num_features[i]], ax=axis)
        i = i + 1
(fig, axes) = plt.subplots(len(_input1[num_features].columns) // 4, 4, figsize=(18, 10))
i = 0
fig.subplots_adjust(hspace=0.5, wspace=0.1)
for triaxis in axes:
    for axis in triaxis:
        sns.boxplot(x=_input1[num_features[i]], ax=axis)
        i = i + 1
categorical = ['HomePlanet', 'VIP', 'CabinDeck', 'Destination', 'CryoSleep', 'CabinSide']
(fig, axes) = plt.subplots(len(_input1[num_features].columns) // 3, 2, figsize=(18, 10))
i = 0
fig.subplots_adjust(hspace=0.5, wspace=0.3)
for triaxis in axes:
    for axis in triaxis:
        sns.barplot(data=_input1, x=_input1[categorical[i]], y='TotalExp', ax=axis, palette='flare')
        i = i + 1
_input1.dtypes
_input1.isnull().sum()
_input0.isnull().sum()
_input1[num_features].isnull().sum()
for col in num_features:
    _input1[col] = _input1[col].fillna(_input1[col].median())
_input1[num_features].isnull().sum()
_input0[num_features].isnull().sum()
for col in num_features:
    _input0[col] = _input0[col].fillna(_input0[col].median())
_input0[num_features].isnull().sum()
_input1[cat_features].isnull().sum()
for col in cat_features:
    _input1[col] = _input1[col].fillna(_input1[col].mode()[0])
_input1['CabinNum'] = _input1['CabinNum'].astype(int)
_input1[cat_features].isnull().sum()
_input0[cat_features].isnull().sum()
for col in cat_features:
    _input0[col] = _input0[col].fillna(_input0[col].mode()[0])
_input0['CabinNum'] = _input0['CabinNum'].astype(int)
_input0[cat_features].isnull().sum()
_input1.isnull().sum()
_input0.isnull().sum()
drop_col = ['Cabin', 'Name']
_input1 = _input1.drop(columns=drop_col)
_input0 = _input0.drop(columns=drop_col)
_input1.head()
_input0.head()
X = _input1.drop(columns=['Transported'])
y = _input1[['PassengerId', 'Transported']]
X.head()
y.head()
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
le = LabelEncoder()
categories = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
encoded_ = []
test_enc = []
for col in categories:
    X_encoded = le.fit_transform(X[col])
    test_encoded = le.fit_transform(_input0[col])
    encoded_.append(X_encoded)
    test_enc.append(test_encoded)
df_encoded = pd.DataFrame({'HomePlanet_': encoded_[0], 'Is_CryoSleep': encoded_[1], 'Destination_': encoded_[2], 'Is_VIP': encoded_[3], 'CabinDeck_': encoded_[4], 'CabinSide_': encoded_[5]})
df_test_encoded = pd.DataFrame({'HomePlanet_': test_enc[0], 'Is_CryoSleep': test_enc[1], 'Destination_': test_enc[2], 'Is_VIP': test_enc[3], 'CabinDeck_': test_enc[4], 'CabinSide_': test_enc[5]})
X = pd.concat([X, df_encoded], axis=1)
X = X.drop(columns=categories, inplace=False)
_input0 = pd.concat([_input0, df_test_encoded], axis=1)
_input0 = _input0.drop(columns=categories, inplace=False)
X.head()
index_ = ['PassengerId']
X = X.set_index(index_)
y = y.set_index(index_)
X.head()
y.head()
_input0.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=101)
log_reg = LogisticRegression(solver='liblinear', C=0.1, max_iter=500)