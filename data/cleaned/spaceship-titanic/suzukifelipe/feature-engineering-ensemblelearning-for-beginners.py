import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_set = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_set = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_set.head()
train_set.describe()
train_set.info()
test_set.info()
df = pd.concat([train_set, test_set])
df.info()
columns_d = df[['Cabin', 'Name', 'Transported']]
target = df[['Transported']]
df = df.drop(columns_d, axis=1)
df.isna().any(axis=1).sum()
print('The total of rows with missing values is: ', round(df.isna().any(axis=1).sum() / len(df.index) * 100, 2), '%')
df_new = df.dropna()
df_new.isna().any(axis=1).sum()
dummies1 = pd.get_dummies(df_new['HomePlanet'])
dummies2 = pd.get_dummies(df_new['Destination'])
df_new = pd.concat([df_new.drop('HomePlanet', axis=1), dummies1], axis=1)
df_new = pd.concat([df_new.drop('Destination', axis=1), dummies2], axis=1)
df_new.head()
df_new['CryoSleep'] = df_new['CryoSleep'].astype(int)
df_new['VIP'] = df_new['VIP'].astype(int)
corr = df_new.corr()
import matplotlib.pyplot as plt
import seaborn as sns
(fig, ax) = plt.subplots(figsize=(15, 10))
sns.heatmap(corr, cmap='Blues', annot=True, linewidths=0.5, ax=ax)
cryo_feat = df_new[['PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
cryo_label = df_new['CryoSleep']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rfc = RandomForestClassifier(random_state=42)
rfc_scores = cross_val_score(rfc, cryo_feat, cryo_label, cv=3)
print('The Mean Accuracy is %0.3f (+/- %0.3f)' % (rfc_scores.mean().mean(), rfc_scores.mean().std() * 2))
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(cryo_feat, cryo_label, test_size=0.3, random_state=42)
(X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=3)