import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
HomePlanet = set(_input1['HomePlanet'])
HomePlanet_stats = {i: np.count_nonzero(_input1['HomePlanet'] == i) for i in _input1['HomePlanet']}
HomePlanet_stats_keys = HomePlanet_stats.keys()
HomePlanet_stats_values = HomePlanet_stats.values()
plt.pie(HomePlanet_stats_values, labels=HomePlanet_stats_keys, autopct='%1.1f%%')
Destination = set(_input1['Destination'])
print(Destination)
Destination_stats = {i: np.count_nonzero(_input1['Destination'] == i) for i in _input1['Destination']}
print(Destination_stats)
Destination = _input1['Destination']
Destination_stats = {i: np.count_nonzero(Destination == i) for i in Destination}
print(Destination_stats)
Destination_stats_keys = Destination_stats.keys()
Destination_stats_values = Destination_stats.values()
plt.pie(Destination_stats_values, labels=Destination_stats.keys(), autopct='%1.1f%%')
Transported = _input1['Transported']
print(Transported.describe())
Transported_stats = {i: np.count_nonzero(Transported == i) for i in Transported}
print(Transported_stats)
Transported_stats_keys = Transported_stats.keys()
Transported_stats_values = Transported_stats.values()
plt.pie(Transported_stats_values, labels=Transported_stats_keys, autopct='%1.1f%%')
Survival_Destination = np.array([_input1['Destination'], _input1['Transported']])
print(Survival_Destination)
test = zip(_input1['Destination'], _input1['Transported'])
Survival_Destination_table = _input1.groupby(['Destination', 'Transported']).size()
print(Survival_Destination_table)
type(Survival_Destination_table)
print(Survival_Destination_table[0:2])
print(Survival_Destination_table.index)
print(Survival_Destination_table.values)
print(len(Survival_Destination_table.index))
print(Survival_Destination_table.reset_index())
Survival_Destination_table_new = Survival_Destination_table.reset_index()
print(type(Survival_Destination_table_new))
print('\n')
print(Survival_Destination_table_new.iloc[1])
print('\n')
print(Survival_Destination_table_new.iloc[0:2])
print('\n')
print(Survival_Destination_table_new.iloc[2])
print('\n')
print(Survival_Destination_table_new.at[2, 'Destination'])
print('\n')
print(Survival_Destination_table_new[0])
print(Survival_Destination_table_new[0][1])
print(Survival_Destination_table_new[0][0:5])
print(Survival_Destination_table_new[0:2])
print(Survival_Destination_table_new[0:2][0])
Survival_Destination_table_new = Survival_Destination_table_new.to_numpy()
(fig, axes) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
axes[0].pie(Survival_Destination_table_new.T[2][0:2], labels=Survival_Destination_table_new.T[1][0:2], autopct='%1.1f%%', frame=True)
axes[0].set_title(Survival_Destination_table_new.T[0][0])
axes[1].pie(Survival_Destination_table_new.T[2][2:4], labels=Survival_Destination_table_new.T[1][2:4], autopct='%1.1f%%', frame=True)
axes[1].set_title(Survival_Destination_table_new.T[0][2])
axes[2].pie(Survival_Destination_table_new.T[2][4:6], labels=Survival_Destination_table_new.T[1][4:6], autopct='%1.1f%%', frame=True)
axes[2].set_title(Survival_Destination_table_new.T[0][4])
plt.plot()
Survival_Origin_table = _input1.groupby(['HomePlanet', 'Transported']).size()
print(Survival_Origin_table)
Survival_Origin_table = Survival_Origin_table.reset_index()
print('\n', Survival_Origin_table)
Survival_Origin_table = Survival_Origin_table.to_numpy()
print('\n', Survival_Origin_table)
print('\n', Survival_Origin_table.T)
(fig, axes) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
axes[0].pie(Survival_Origin_table.T[2][0:2], autopct='%1.1f%%', labels=Survival_Origin_table.T[1][0:2])
axes[1].pie(Survival_Origin_table.T[2][2:4], autopct='%1.1f%%', labels=Survival_Origin_table.T[1][2:4])
axes[2].pie(Survival_Origin_table.T[2][4:6], autopct='%1.1f%%', labels=Survival_Origin_table.T[1][4:6])
axes[0].set_title(Survival_Origin_table.T[0][1])
axes[1].set_title(Survival_Origin_table.T[0][3])
axes[2].set_title(Survival_Origin_table.T[0][5])
plt.plot()
_input1.head()
Multi_table = _input1.groupby(['CryoSleep', 'Age', 'VIP', 'Transported']).size()
print(Multi_table)
Multi_table = _input1.groupby(['VIP', 'CryoSleep', 'Transported']).size()
print('\n', Multi_table)
Multi_table = _input1.groupby(['VIP', 'Transported']).size()
print('\n', Multi_table)
Multi_table = _input1.groupby(['CryoSleep', 'Transported']).size()
print('\n', Multi_table)
Multi_table = _input1.groupby(['CryoSleep', 'VIP', 'Transported']).size()
print('\n', Multi_table)
AgeDist = _input1['Age']
print(AgeDist)
plt.hist(AgeDist, bins=40, histtype='step', linewidth=5, density=True)
plt.ylabel('Count / Percentage')
plt.xlabel('Age')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
_input1.head()
data_clean = _input1.fillna(0)
features = ['CryoSleep']
data_features = data_clean[features]
data_features = data_features
print(data_features.head())
print('\n', type(data_features))
target = data_clean['Transported']
print('\n', target.head())
(train_features, val_features, train_target, val_target) = train_test_split(data_features, target, random_state=0)
forest_model = RandomForestClassifier(random_state=1)