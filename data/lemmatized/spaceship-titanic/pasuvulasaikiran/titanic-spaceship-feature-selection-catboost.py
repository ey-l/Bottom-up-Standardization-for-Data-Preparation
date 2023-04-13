import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input0
_input1.isnull().sum()
_input1.isnull().mean() * 100
_input0.isnull().mean() * 100
_input1['isTrain'] = 'Yes'
_input0['isTrain'] = 'No'
tt = pd.concat([_input1.drop('Transported', axis=1), _input0])
tt
tt[['Cabin1_', 'Cabin2_', 'Cabin3_']] = tt['Cabin'].str.split('/', expand=True)
tt
tt[['Pid1_', 'Pid2_']] = tt['PassengerId'].str.split('_', expand=True).astype('int')
tt
tt[['Fname_', 'Lname_']] = tt['Name'].str.split(' ', expand=True)
tt
tt['sum_exp_'] = tt['RoomService'] + tt['FoodCourt'] + tt['ShoppingMall'] + tt['Spa'] + tt['VRDeck']
tt
tt['mean_exp_'] = tt['sum_exp_'] / tt['Pid2_']
tt
tt['Age_cat_'] = pd.cut(tt.Age, bins=[0, 5, 12, 18, 50, 150], labels=['Toddler/Baby', 'Child', 'Teen', 'Adult', 'Elderly'])
tt['Age_cat_']
numerical = tt.select_dtypes(exclude=['object', 'category']).columns.to_list()
numerical
for i in tt.columns:
    print('{} ------------------------------------> {}'.format(i, tt[i].nunique()))
tt.isnull().mean() * 100
tt = tt.set_index('PassengerId')
tt
categorical = tt.select_dtypes(['object', 'category']).columns.to_list()
categorical
numerical = tt.select_dtypes(exclude=['object', 'category']).columns.to_list()
numerical
tt[numerical]
tt[numerical].columns
columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'sum_exp_', 'mean_exp_']
(q, r) = divmod(len(columns), 2)
(fig, ax) = plt.subplots(q, 2, figsize=(18, 10))
for i in range(0, len(columns)):
    (q, r) = divmod(i, 2)
    sns.kdeplot(data=tt[numerical], x=columns[i], ax=ax[q, r])
columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'sum_exp_', 'mean_exp_']
(q, r) = divmod(len(columns), 2)
(fig, ax) = plt.subplots(q, 2, figsize=(18, 10))
for i in range(0, len(columns)):
    (q, r) = divmod(i, 2)
    sns.boxplot(data=tt[numerical], x=columns[i], ax=ax[q, r])
from feature_engine.imputation import MeanMedianImputer
median_imputer = MeanMedianImputer(imputation_method='median', variables=numerical)