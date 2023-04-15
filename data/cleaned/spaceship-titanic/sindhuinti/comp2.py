import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.isnull().sum()
train_data = train_data.drop(['Cabin', 'Name'], axis=1)
test_data = test_data.drop(['Cabin', 'Name'], axis=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_data = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)
train_data['CryoSleep'] = train_data['CryoSleep'].astype('int')
test_data['CryoSleep'] = test_data['CryoSleep'].astype('int')
train_data['VIP'] = train_data['VIP'].astype('int')
test_data['VIP'] = test_data['VIP'].astype('int')
train_data['Transported'] = train_data['Transported'].astype('int')
train_data.head()
train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'Destination'])
test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'Destination'])
train_data.isnull().sum()
X_train = train_data.drop(['Transported'], axis=1)
y_train = train_data['Transported']
X_test = test_data
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()