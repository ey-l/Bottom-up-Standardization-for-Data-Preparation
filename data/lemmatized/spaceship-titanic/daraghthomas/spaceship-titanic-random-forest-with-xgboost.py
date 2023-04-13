import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
df = pd.concat([_input1, _input0])
df.info()
print(df.head())
for column in df.columns:
    missing_count = df[column].isnull().sum()
    if missing_count > 0:
        print(f"Column '{column}' has {missing_count} missing values")
cat_data = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
for cat in cat_data:
    df[cat] = df[cat].fillna(df[cat].mode().iloc[0], inplace=False)
num_data = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for num in num_data:
    df[num] = df[num].fillna(df[num].median(), inplace=False)
for column in df.columns:
    missing_count = df[column].isnull().sum()
    if missing_count > 0:
        print(f"Column '{column}' has {missing_count} missing values")
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for column in spend_cols:
    print(df[column].isnull().sum())
for column in spend_cols:
    column_sum = df[column].sum()
    formatted_string = '${:,.0f}'.format(column_sum)
    print(f'{df[column].name}: {formatted_string}')
df['Total_Spending'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
total_spent_str = '${:,.0f}'.format(df['Total_Spending'].sum())
mean_spent_str = '${:,.2f}'.format(df['Total_Spending'].mean())
median_spent_str = '${:,.0f}'.format(df['Total_Spending'].median())
mode_spent_str = '${:,.0f}'.format(df['Total_Spending'].mode().iloc[0])
most_spent_str = '${:,.0f}'.format(df['Total_Spending'].max())
print('Total Spent:', total_spent_str)
print('Mean amount spent by each passenger:', mean_spent_str)
print('Median amount spent by each passenger:', median_spent_str)
print('Mode amount spent by each passenger:', mode_spent_str)
print('Most amount spent:', most_spent_str)
train_df = df[df['Transported'].notnull()]
test_df = df[df['Transported'].isnull()]
input_features = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_Spending']
target_variable = 'Transported'
cat_features = ['HomePlanet', 'Cabin', 'Destination']
train_df.loc[:, cat_features] = train_df[cat_features].astype('category')
test_df.loc[:, cat_features] = test_df[cat_features].astype('category')
print(test_df.info())
print('XGBoost version:', xgb.__version__)
X = train_df[input_features]
y = train_df[target_variable]
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=42)
y_val = y_val.astype(int)
model = xgb.XGBClassifier()