import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_df.head()
test_df.describe()
test_df.describe(include=object)
test_df.isna().sum()
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_df.head()
train_df.describe()
train_df.describe(include=object)
train_df.isna().sum()
vip = train_df.loc[train_df.VIP == True]['Transported']
rate_vip = sum(vip) / len(vip)
print(f'VIP who survived: {rate_vip}')
no_vip = train_df.loc[train_df.VIP == False]['Transported']
rate_no_vip = sum(no_vip) / len(no_vip)
print(f'VIP who survived: {rate_no_vip}')
from sklearn.ensemble import RandomForestClassifier
y = train_df['Transported']
features = ['VIP', 'CryoSleep', 'HomePlanet']
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)