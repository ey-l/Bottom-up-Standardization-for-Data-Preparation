import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
sns.set(color_codes=True)
np.random.seed(42)
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Raw data:')
train_df.head()
train_df.hist(figsize=(15, 10))
train_df = train_df.drop(labels=['PassengerId', 'Cabin', 'Name'], axis=1)
test_passenger_ids = test_df['PassengerId']
test_df = test_df.drop(labels=['PassengerId', 'Cabin', 'Name'], axis=1)
print(test_df.head())
train_df.head()
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
num_imputer = SimpleImputer(strategy='mean')
train_df[numerical_cols] = pd.DataFrame(num_imputer.fit_transform(train_df[numerical_cols]), columns=numerical_cols)
test_df[numerical_cols] = pd.DataFrame(num_imputer.fit_transform(test_df[numerical_cols]), columns=numerical_cols)
print('NaN value count in each column:')
train_df.isna().sum()
class_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
cls_imputer = SimpleImputer(strategy='most_frequent')
train_df[class_cols] = pd.DataFrame(cls_imputer.fit_transform(train_df[class_cols]), columns=class_cols)
test_df[class_cols] = pd.DataFrame(cls_imputer.fit_transform(test_df[class_cols]), columns=class_cols)
print('NaN value count in each column:')
train_df.isnull().sum()
class_cols.append('Transported')
for col in class_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
class_cols.remove('Transported')
for col in class_cols:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])
train_df.head()
(fig, ax) = plt.subplots(figsize=(15, 10))
sns.heatmap(train_df.corr(), annot=True, ax=ax)
(X_train, X_valid, y_train, y_valid) = train_test_split(train_df.loc[:, train_df.columns != 'Transported'], train_df[['Transported']], test_size=0.1)
print('Training dataset shapes:', X_train.shape, y_train.shape)
print('Validation dataset shapes:', X_valid.shape, y_valid.shape)
tree = DecisionTreeClassifier(random_state=42)