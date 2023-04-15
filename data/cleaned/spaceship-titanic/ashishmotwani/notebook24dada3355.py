import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train = df.drop(['Cabin', 'Name'], axis=1)
df = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test = df.drop(['Cabin', 'Name'], axis=1)
df_train
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df_train['HomePlanet'] = label.fit_transform(df_train['HomePlanet'])
df_test['HomePlanet'] = label.transform(df_test['HomePlanet'])
df_train['CryoSleep'] = label.fit_transform(df_train['CryoSleep'])
df_test['CryoSleep'] = label.transform(df_test['CryoSleep'])
df_train['Destination'] = label.fit_transform(df_train['Destination'])
df_test['Destination'] = label.transform(df_test['Destination'])
df_train['VIP'] = label.fit_transform(df_train['VIP'])
df_test['VIP'] = label.transform(df_test['VIP'])
df_train['Transported'] = label.fit_transform(df_train['Transported'])
y = df_train['Transported']
df_test = df_test.drop(['PassengerId'], axis=1)
df_train = df_train.drop(['PassengerId', 'Transported'], axis=1)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_train = pd.DataFrame(my_imputer.fit_transform(df_train))
X_test = pd.DataFrame(my_imputer.transform(df_test))
X_train.columns = df_train.columns
X_test.columns = df_test.columns
X_train.isnull().sum()
X_test.isnull().sum()
from sklearn.ensemble import GradientBoostingClassifier
model_gbc = GradientBoostingClassifier(n_estimators=199)