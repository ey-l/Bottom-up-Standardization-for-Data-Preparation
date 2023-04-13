import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_types = {'HomePlanet': 'category', 'Destination': 'category'}
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv', dtype=data_types)
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv', dtype=data_types)
_input1.head(10)
_input1 = _input1.replace({True: 1, False: 0})
_input0 = _input0.replace({True: 1, False: 0})
_input1.head(10)
_input1.info()
_input1.describe()
_input1.Transported.value_counts()
plt.subplots(figsize=(6, 5))
plt.pie(_input1.Transported.value_counts(normalize=True), labels=['Transported', 'Not transported'], startangle=90, autopct='%.2f%%')
plt.title('Distribution of the target')
(fig, axes) = plt.subplots(1, 2, figsize=(15, 6))
b = sns.countplot(data=_input1, x='HomePlanet', hue='Transported', ax=axes[0]).set_title('Number of passengers per Home Planet')
c = sns.barplot(data=_input1, x='HomePlanet', y='Transported', ax=axes[1], ci=None)
c.set_xlabel('Number of passengers')
c.set_title('Ratio of transported passengers per Home planet')
sns.despine()
f = sns.catplot(data=_input1, x='CryoSleep', hue='Transported', kind='count')
f.set(title='Number of passengers in CryoSleep').set_axis_labels('CryoSleep status', 'No of passsengers').set_xticklabels(['Not in sleep', 'Sleep']).despine()
g = sns.catplot(data=_input1, x='CryoSleep', y='Transported', kind='bar', ci=None)
g.despine().set(title='Ratio of transported (lost) persons per CryoSleep status').set_xticklabels(['Awake', 'Sleeping'])
g.set_axis_labels('CryoSleep status', 'Ratio')
dest = list(_input1.Destination.unique())
print(f'Destinations of the passengers: {dest[0]}, {dest[1]}, {dest[2]}')
h = sns.catplot(data=_input1, x='Destination', y='Transported', kind='bar', ci=None)
h.despine().set(title='Ratio of passengers lost per Destination').set_axis_labels('Destination', 'Ratio of lost')
i = sns.catplot(data=_input1, x='VIP', y='Transported', kind='bar', ci=None)
i.despine().set(title='VIP status and chanches of getting lost').set_axis_labels('VIP status', 'Ratio of getting lost')
i.set_xticklabels(['Not VIP', 'VIP'])
(fig, ax) = plt.subplots(1, 1, figsize=(10, 5))
g = sns.histplot(data=_input1, x='Age', hue='Transported', bins=79, ax=ax)
g.set(title='Distribution of age of lost and not lost passengers')
g.set_ylabel('Number of passengers')
plt.legend(labels=['Lost', 'Not lost'], title='Got lost')
sns.despine()
age = _input1.groupby('Age')['Transported'].agg([np.mean, sum]).reset_index()
(fig, ax) = plt.subplots(1, 1, figsize=(10, 8))
j = sns.scatterplot(data=age, x='Age', y='mean', size='sum', ax=ax)
j.set(title='Ratio of lost passengers per age')
j.set_ylabel('Ratio of lost passengers')
plt.legend(title='Amount of people lost')
sns.despine()
long = _input1.melt(id_vars=['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'Name', 'Transported'], var_name='Services', value_name='Spending')
long
(fig, ax) = plt.subplots(1, 1, figsize=(20, 8))
k = sns.boxplot(data=long, y='Services', x='Spending', hue='Transported')
plt.legend(title='Transported')
plt.title('Spendings depending on transported status')
sns.despine()
_input1['Total_spending'] = _input1.RoomService + _input1.FoodCourt + _input1.ShoppingMall + _input1.Spa + _input1.VRDeck
_input0['Total_spending'] = _input0.RoomService + _input0.FoodCourt + _input0.ShoppingMall + _input0.Spa + _input0.VRDeck
_input1.head()
(fig, ax) = plt.subplots(1, 1, figsize=(20, 5))
o = sns.histplot(data=_input1, x='Total_spending', hue='Transported', ax=ax)
sns.despine()
p = sns.boxplot(data=_input1, x='Transported', y='Total_spending')
p.set_xticklabels(['Not transported', 'Transported'])
p.set_title('Distribution of spending')
sns.despine()
_input1['Family_group'] = _input1.PassengerId.str[:4]
_input0['Family_group'] = _input0.PassengerId.str[:4]
_input1.head(10)
train_fam_groups = _input1.Family_group.unique()
test_fam_groups = _input0.Family_group.unique()
overlapping_fam_groups = [fam_group for fam_group in train_fam_groups if fam_group in test_fam_groups]
_input1['Family_size'] = _input1.groupby('Family_group')['Family_group'].transform('count')
_input0['Family_size'] = _input0.groupby('Family_group')['Family_group'].transform('count')
_input1.head(20)
(fig, ax) = plt.subplots(1, 1, figsize=(8, 5))
l = sns.countplot(data=_input1, x='Family_size', hue='Transported', ax=ax).set(title='Distribution of family sizes according to transported(lost) and not transported passengers')
sns.despine()
m = sns.catplot(data=_input1, x='Family_size', y='Transported', kind='bar', ci=None)
m.set(title='Rate of getting lost per family sizes')
_input1['Deck'] = _input1.Cabin.str[0]
_input0['Deck'] = _input0.Cabin.str[0]
plot_order = _input1.groupby('Deck')['Transported'].mean().sort_values(ascending=True).index
plot_order
n = sns.catplot(data=_input1, x='Deck', y='Transported', kind='bar', ci=None, order=plot_order)
n.set(title='Ratio of passengers transported per deck')
deck_encoding_dict = {'T': 1, 'E': 2, 'D': 2, 'F': 2, 'A': 2, 'G': 2, 'C': 3, 'B': 3}
deck_encoding_dict
_input1['Deck'] = _input1.Deck.replace(deck_encoding_dict)
_input0['Deck'] = _input0.Deck.replace(deck_encoding_dict)
_input1.head()
_input0.head()
_input1['Side'] = _input1.Cabin.str[-1]
_input0['Side'] = _input0.Cabin.str[-1]
o = sns.catplot(data=_input1, x='Side', y='Transported', ci=None, kind='bar')
o.set(title='Ratio of transported people per side')
mask = np.triu(_input1.corr(), k=0)
plt.figure(figsize=(10, 10))
r = sns.heatmap(_input1.corr(), annot=True, mask=mask, cmap='coolwarm').set_title('Correlation of the features in the training set')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
_input1.dropna()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Deck', 'Family_size']
cat_cols = ['HomePlanet', 'Destination', 'Side']
X = _input1[num_cols + cat_cols].copy()
y = _input1['Transported']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.25, random_state=42)
logreg = LogisticRegression(random_state=42)
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state=42)
extra_trees = ExtraTreesClassifier(random_state=42)
gbc = GradientBoostingClassifier(learning_rate=0.08, n_estimators=80, max_depth=4, random_state=42)
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])
numerical_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
preprocessor = ColumnTransformer([('cat', categorical_transformer, cat_cols), ('num', numerical_transformer, num_cols)])
pipe = Pipeline([('preproc', preprocessor), ('classifier', gbc)])
param_grid = {'classifier__n_estimators': [80], 'classifier__max_depth': [4], 'classifier__learning_rate': [0.08]}
search = GridSearchCV(pipe, param_grid, refit=True, verbose=3, scoring='accuracy')