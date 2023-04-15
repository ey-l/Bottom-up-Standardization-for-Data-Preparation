import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
train_ = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_ = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_.head()
train_.shape
train_.columns.tolist()
train_.dtypes
train_['TotalExp'] = train_['RoomService'] + train_['FoodCourt'] + train_['ShoppingMall'] + train_['Spa'] + train_['VRDeck']
test_['TotalExp'] = test_['RoomService'] + test_['FoodCourt'] + test_['ShoppingMall'] + test_['Spa'] + test_['VRDeck']
train_.head()
train_[['CabinDeck', 'CabinNum', 'CabinSide']] = train_['Cabin'].str.split('/', expand=True)
test_[['CabinDeck', 'CabinNum', 'CabinSide']] = test_['Cabin'].str.split('/', expand=True)
train_.head()
train_[['PassengerGrp', 'PassengerNum']] = train_['PassengerId'].str.split('_', expand=True).astype(int)
test_[['PassengerGrp', 'PassengerNum']] = test_['PassengerId'].str.split('_', expand=True).astype(int)
train_.head()
for col in train_.columns:
    print(f'{col} ---------> {train_[col].nunique()}')
for col in test_.columns:
    print(f'{col} ---------> {test_[col].nunique()}')
num_features = train_.select_dtypes(exclude=['object', 'bool']).columns.tolist()
train_[num_features].head()
cat_features = train_.select_dtypes(include=['object']).columns.tolist()
train_[cat_features].head()
train_.describe()
transported_df = train_.copy()
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
        sns.countplot(data=transported_df, x=train_[feat_cat[i]], hue='CryoSleep', palette='flare', ax=axis)
        i = i + 1
(fig, axes) = plt.subplots(len(train_[num_features].columns) // 4, 4, figsize=(18, 10))
i = 0
fig.subplots_adjust(hspace=0.5, wspace=0.5)
for triaxis in axes:
    for axis in triaxis:
        sns.kdeplot(data=train_[num_features[i]], ax=axis)
        i = i + 1
(fig, axes) = plt.subplots(len(train_[num_features].columns) // 4, 4, figsize=(18, 10))
i = 0
fig.subplots_adjust(hspace=0.5, wspace=0.1)
for triaxis in axes:
    for axis in triaxis:
        sns.boxplot(x=train_[num_features[i]], ax=axis)
        i = i + 1
categorical = ['HomePlanet', 'VIP', 'CabinDeck', 'Destination', 'CryoSleep', 'CabinSide']
(fig, axes) = plt.subplots(len(train_[num_features].columns) // 3, 2, figsize=(18, 10))
i = 0
fig.subplots_adjust(hspace=0.5, wspace=0.3)
for triaxis in axes:
    for axis in triaxis:
        sns.barplot(data=train_, x=train_[categorical[i]], y='TotalExp', ax=axis, palette='flare')
        i = i + 1
train_.dtypes
train_.isnull().sum()
test_.isnull().sum()
train_[num_features].isnull().sum()
for col in num_features:
    train_[col] = train_[col].fillna(train_[col].median())
train_[num_features].isnull().sum()
test_[num_features].isnull().sum()
for col in num_features:
    test_[col] = test_[col].fillna(test_[col].median())
test_[num_features].isnull().sum()
train_[cat_features].isnull().sum()
for col in cat_features:
    train_[col] = train_[col].fillna(train_[col].mode()[0])
train_['CabinNum'] = train_['CabinNum'].astype(int)
train_[cat_features].isnull().sum()
test_[cat_features].isnull().sum()
for col in cat_features:
    test_[col] = test_[col].fillna(test_[col].mode()[0])
test_['CabinNum'] = test_['CabinNum'].astype(int)
test_[cat_features].isnull().sum()
train_.isnull().sum()
test_.isnull().sum()
drop_col = ['Cabin', 'Name']
train_ = train_.drop(columns=drop_col)
test_ = test_.drop(columns=drop_col)
train_.head()
test_.head()
X = train_.drop(columns=['Transported'])
y = train_[['PassengerId', 'Transported']]
X.head()
y.head()
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
le = LabelEncoder()
categories = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
encoded_ = []
test_enc = []
for col in categories:
    X_encoded = le.fit_transform(X[col])
    test_encoded = le.fit_transform(test_[col])
    encoded_.append(X_encoded)
    test_enc.append(test_encoded)
df_encoded = pd.DataFrame({'HomePlanet_': encoded_[0], 'Is_CryoSleep': encoded_[1], 'Destination_': encoded_[2], 'Is_VIP': encoded_[3], 'CabinDeck_': encoded_[4], 'CabinSide_': encoded_[5]})
df_test_encoded = pd.DataFrame({'HomePlanet_': test_enc[0], 'Is_CryoSleep': test_enc[1], 'Destination_': test_enc[2], 'Is_VIP': test_enc[3], 'CabinDeck_': test_enc[4], 'CabinSide_': test_enc[5]})
X = pd.concat([X, df_encoded], axis=1)
X.drop(columns=categories, inplace=True)
test_ = pd.concat([test_, df_test_encoded], axis=1)
test_.drop(columns=categories, inplace=True)
X.head()
index_ = ['PassengerId']
X = X.set_index(index_)
y = y.set_index(index_)
X.head()
y.head()
test_.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=101)
log_reg = LogisticRegression(solver='liblinear', C=0.1, max_iter=500)