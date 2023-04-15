
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('data/input/spaceship-titanic/train.csv')
df_train
df_test = pd.read_csv('data/input/spaceship-titanic/test.csv')
df_test
df_train.isnull().sum()
df_train.isnull().mean() * 100
df_test.isnull().mean() * 100
df_train['isTrain'] = 'Yes'
df_test['isTrain'] = 'No'
tt = pd.concat([df_train.drop('Transported', axis=1), df_test])
tt
tt[['Cabin1_', 'Cabin2_', 'Cabin3_']] = tt['Cabin'].str.split('/', expand=True)
tt
tt[['Pid1_', 'Pid2_']] = tt['PassengerId'].str.split('_', expand=True).astype('int')
tt
tt[['Fname_', 'LName_']] = tt['Name'].str.split(' ', expand=True)
tt
tt['Age_bin'] = pd.cut(x=tt['Age'], bins=[0, 3, 12, 19, 40, 60, 150], labels=[1, 2, 3, 4, 5, 6])
tt['Age_bin']
for i in tt.columns:
    print('{} ------------------------------------> {}'.format(i, tt[i].nunique()))
tt.isnull().mean() * 100
tt = tt.set_index('PassengerId')
tt
for i in tt.columns:
    print('{} ------------------------------------> {}'.format(i, tt[i].nunique()))
categorical = list(tt.select_dtypes(include=['category', object]).columns)
categorical
numerical = list(tt.select_dtypes(exclude=['category', object]).columns)
numerical
tt
tt[numerical]
tt[numerical].columns
columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(q, r) = divmod(len(columns), 2)
(fig, ax) = plt.subplots(q, 2, figsize=(15, 10))
for i in range(0, len(columns)):
    (q, r) = divmod(i, 2)
    sns.kdeplot(data=tt[numerical], x=columns[i], ax=ax[q, r])

columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
(q, r) = divmod(len(columns), 2)
(fig, ax) = plt.subplots(q, 2, figsize=(15, 10))
for i in range(0, len(columns)):
    (q, r) = divmod(i, 2)
    sns.boxplot(data=tt[numerical], x=columns[i], ax=ax[q, r])

from feature_engine.imputation import MeanMedianImputer
median_imputer = MeanMedianImputer(imputation_method='median', variables=numerical)