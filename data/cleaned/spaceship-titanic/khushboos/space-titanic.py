import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
submission_data = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(f'Train Data\n{train_data.head(5)}\n\nTest Data\n{test_data.head(5)}\nsubmission Data\n\n{submission_data.head(5)}')
print(f'Training data Null Values :\n\n{train_data.isnull().sum()}\n\nShape :\n{train_data.shape}')
drop_col = ['Name', 'Cabin']
train_data.drop(drop_col, axis=1, inplace=True)
test_data.drop(drop_col, axis=1, inplace=True)
plt.subplots(1, 2, figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(train_data.isnull(), interpolation='nearest', cmap='Blues', aspect='auto')
plt.xlabel('Columns of Train data')
plt.ylabel('Values index no of the Null values')
plt.title('Heatmap for the null values')
plt.subplot(1, 2, 2)
train_data.isnull().mean().plot(kind='bar', title='Mean of the Null values', ylabel='Missing values Ratio')

train_data.info()
count_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in train_data[count_col]:
    print(train_data[col].value_counts())
    print('\n')
    print(train_data[col].mode())
    print('\n--------------------------')
for col in count_col:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    test_data[col].fillna(test_data[col].mode()[0], inplace=True)
train_data.isnull().sum()
sns.violinplot(train_data['Age'], origin='h')
plt.title('Age Distribution')
print(f"Median Age:\t{train_data['Age'].median()}")
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
sns.violinplot(data=train_data, orient='h')
Spendings = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in Spendings:
    train_data[col].fillna(train_data[col].median(), inplace=True)
    test_data[col].fillna(test_data[col].median(), inplace=True)
train_data.head(5)
print(f"{train_data['HomePlanet'].unique()}\n{train_data['Destination'].unique()}")
diff_col = ['HomePlanet', 'Destination']
train_data = pd.concat([train_data, pd.get_dummies(train_data[diff_col])], axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data[diff_col])], axis=1)
train_data.drop(diff_col, axis=1, inplace=True)
test_data.drop(diff_col, axis=1, inplace=True)
train_data.info()
col = ['VIP', 'CryoSleep']
train_data[col] = train_data[col].astype('int')
test_data[col] = test_data[col].astype('int')
train_data.info()
train_Y = train_data['Transported']
train_X = train_data.drop(columns=['Transported'], axis=1)
print(train_X.shape, train_Y.shape)
(X, X_val, Y, Y_val) = train_test_split(train_X, train_Y, train_size=0.8, random_state=42)
print(X.shape, X_val.shape, Y.shape, Y_val.shape)
RF = RandomForestClassifier(max_depth=9.5)