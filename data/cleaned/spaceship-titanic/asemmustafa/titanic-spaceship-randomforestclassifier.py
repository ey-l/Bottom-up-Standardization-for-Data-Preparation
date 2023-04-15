import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv', index_col='PassengerId')
train_data.head()
train_data.tail()
train_data.describe().T
train_data = train_data.drop('Name', axis=1)
train_data.isnull().sum()
train_data.dtypes
import seaborn as sns
pp = train_data.corr()
sns.heatmap(pp)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['Transported'] = le.fit_transform(train_data['Transported'])
train_data['HomePlanet'] = le.fit_transform(train_data['HomePlanet'])
train_data['Cabin'] = le.fit_transform(train_data['Cabin'])
train_data['Destination'] = le.fit_transform(train_data['Destination'])
train_data['CryoSleep'] = le.fit_transform(train_data['CryoSleep'])
train_data['VIP'] = le.fit_transform(train_data['VIP'])
import warnings
warnings.filterwarnings('ignore')
count = 1
plt.subplots(figsize=(20, 15))
for i in train_data.columns:
    if train_data[i].dtypes != 'object':
        plt.subplot(6, 7, count)
        sns.distplot(train_data[i])
        count += 1

for i in train_data.columns:
    if train_data[i].dtype == 'object':
        train_data[i].fillna(train_data[i].mode()[0], inplace=True)
    else:
        train_data[i].fillna(train_data[i].median(), inplace=True)
print(train_data.isnull().sum())
y = train_data['Transported']
x = train_data.drop('Transported', axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model = RandomForestClassifier(max_depth=10, random_state=42)
(train_X, test_x, train_y, test_y) = train_test_split(x, y, test_size=0.25, random_state=42)