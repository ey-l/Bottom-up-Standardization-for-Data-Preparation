

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.impute import KNNImputer
import xgboost as xgb
sns.set_theme()
sns.set(font_scale=1.2)
plt.style.use('seaborn-whitegrid')
palette = None
path = 'data/input/spaceship-titanic/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
train.head()
print('Train dataset shape:', train.shape)
print('Test dataset shape:', test.shape)
missing = round(100 * train.isnull().sum() / train.shape[0], 2)
missing_df = pd.DataFrame({'% of missing values': missing})
missing_df
(fig, ax1) = plt.subplots(figsize=(7, 3))
sns.countplot(data=train, y='Transported', ax=ax1, orient='h', palette=palette)
sns.despine()

transported_perc = 100 * (train['Transported'] == True).sum() / (train['Transported'] == True).count()
print('The overall percentage of transported passengers is: {:.1f}%'.format(transported_perc))

def plot_categorical_feature(feature, data=train):
    (fig, axs) = plt.subplots(1, 2, figsize=(9, 4))
    fig.subplots_adjust(hspace=0.3)
    values = list(pd.Series(data[feature].unique()).sort_values())
    if np.nan in values:
        values.remove(np.nan)
    sns.barplot(data=data, x=feature, y='Transported', order=values, ax=axs[0], palette=palette)
    sns.despine()
    axs[0].set_xlabel('')
    if feature == 'Destination':
        axs[0].xtick_params(axis='x', labelrotation=10)
    axs[0].set_title(feature)
    sizes = []
    for value in values:
        sizes.append(data.loc[data[feature] == value, feature].size)
    explode = []
    for size in sizes:
        if size == max(sizes):
            explode.append(0.06)
        else:
            explode.append(0)
    (w, _, autotexts) = axs[1].pie(sizes, explode=explode, shadow=True, autopct='%1.0f%%', colors=palette)
(fig, axs) = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=train, x='HomePlanet', hue='Transported', ax=axs[0])
axs[0].set_title('HomePlanet', fontweight='bold')
axs[0].set_xlabel('')
sns.despine()
sns.countplot(data=train, x='CryoSleep', hue='Transported', ax=axs[1])
axs[1].set_title('CryoSleep', fontweight='bold')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
sns.despine()
(fig, axs) = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=train, x='Destination', hue='Transported', ax=axs[0])
axs[0].set_title('Destination', fontweight='bold')
axs[0].set_xlabel('')
sns.despine()
sns.countplot(data=train, x='VIP', hue='Transported', ax=axs[1])
axs[1].set_title('VIP', fontweight='bold')
axs[1].set_xlabel('')
axs[1].set_ylabel('')
sns.despine()
train['Cabin'].head()
visual_data = {'Deck': list(train['Cabin'].str.split('/', expand=True)[0].copy()), 'DeckSide': list(train['Cabin'].str.split('/', expand=True)[2].copy()), 'Transported': list(train['Transported'].copy())}
visual = pd.DataFrame(visual_data)
(fig, axs) = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=visual, x='Deck', y='Transported', ax=axs[0])
sns.despine()
sns.countplot(data=visual, x='Deck', ax=axs[1])
sns.despine()


def bool_cabin(value):
    if value == 'B' or value == 'C':
        return 1
    elif value in ['F', 'A', 'G', 'E', 'D', 'T']:
        return 0
    else:
        return value
visual['Deck_BC'] = visual['Deck'].apply(bool_cabin)
features = ['Deck_BC']
visual = visual.dropna()
plot_categorical_feature('Deck_BC', data=visual)

plot_categorical_feature('DeckSide', data=visual)
visual = train.copy()
visual['CabinNumber'] = visual['Cabin'].str.split('/', expand=True)[1].copy()
visual = visual[~visual['CabinNumber'].isnull()].copy()
visual['CabinNumber'] = visual['CabinNumber'].astype('int32')
visual['CabinNumber'].unique()
labels = ['A', 'B', 'C', 'D', 'E', 'F']
visual['CabinGroup'] = pd.cut(visual['CabinNumber'], 6, labels=labels)
plot_categorical_feature('CabinGroup', data=visual)

def extract_cabin_group(value):
    if value in ['A', 'C', 'D']:
        return 1
    elif value in ['B', 'E', 'F']:
        return 0
    else:
        return value
visual['CabinGroup'] = visual['CabinGroup'].apply(extract_cabin_group)
plot_categorical_feature('CabinGroup', data=visual)
train['PassengerId'].head()
values = list(train['PassengerId'].str.split('_', expand=True)[0].copy())
group_data = {'Pgroup': values}
visual = pd.DataFrame(group_data)
visual['GroupSize'] = 0
visual['Transported'] = train['Transported'].copy()
groups = list(visual.groupby('Pgroup')['Pgroup'].groups.keys())
values = list(visual.groupby('Pgroup')['Pgroup'].count())
for (group, value) in zip(groups, values):
    visual.loc[visual['Pgroup'] == group, 'GroupSize'] = visual.loc[visual['Pgroup'] == group, 'GroupSize'].apply(lambda x: value)
(fig, axs) = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=visual, x='GroupSize', y='Transported', ax=axs[0])
sns.despine()
sns.countplot(data=visual, x='GroupSize', ax=axs[1])
sns.despine()


def group_class(value):
    if value in [3, 4, 5, 6]:
        return 1
    else:
        return 0
visual['GroupSize_M'] = visual['GroupSize'].apply(group_class)
plot_categorical_feature('GroupSize_M', data=visual)

(fig, ax) = plt.subplots(figsize=(9, 5))
sns.histplot(data=train, x='Age', hue='Transported', bins=20, kde=True, ax=ax, palette=palette)

train[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].describe()
sns.histplot(data=train[train['RoomService'] < 200], x='RoomService', hue='Transported', kde=True)


def no_exp(value):
    if value == 0:
        return 1
    elif value >= 0:
        return 0
    else:
        return value

def plot_numerical_features(features):
    (fig, axs) = plt.subplots(2, 3, figsize=(16, 8))
    i = 0
    for feature in features:
        mask = visual[feature] != 0
        sns.histplot(data=visual[mask], x=np.log(1 + visual.loc[mask, feature]), hue='Transported', kde=True, ax=axs[0, i], palette=palette)
        sns.despine()
        axs[0, i].set_title(feature)
        axs[0, i].set_xlabel('')
        if i != 1:
            axs[0, i].legend('')
        if i != 0:
            axs[0, i].set_ylabel('')
        visual[feature + '_NoExp'] = visual[feature].apply(no_exp)
        sns.countplot(data=visual, x=visual[feature + '_NoExp'], hue='Transported', ax=axs[1, i], alpha=0.95, palette=palette)
        sns.despine()
        if i != 1:
            axs[1, i].legend('')
        if i != 0:
            axs[1, i].set_ylabel('')
        else:
            axs[1, i].set_ylabel('Count')
        i += 1
visual = train.copy()
visual['TotalExpenses'] = visual['RoomService'] + visual['Spa'] + visual['VRDeck'] + visual['FoodCourt'] + visual['ShoppingMall']
features = ['RoomService', 'Spa', 'VRDeck']
plot_numerical_features(features)
log_scale = list(np.linspace(2, 10, 5))
original_scale = []
for l in log_scale:
    original_scale.append('{:.0f}'.format(np.exp(l) - 1))
pd.DataFrame({'log_scale': log_scale, 'original_scale': original_scale})
features = ['FoodCourt', 'ShoppingMall', 'TotalExpenses']
plot_numerical_features(features)
mask = visual['CryoSleep'] == False
sns.countplot(data=visual[mask], x='TotalExpenses_NoExp', hue='Transported', alpha=0.95, palette=palette)
sns.despine()
sns.color_palette()
sns.relplot(data=visual, x='Age', y='TotalExpenses', hue='HomePlanet', col='Destination', height=6, aspect=0.8)
sns.despine()

X_train = train.iloc[:, 0:-1].copy()
y_train = train.iloc[:, -1].copy()
train_id = X_train['PassengerId']
test_id = test['PassengerId']
data = pd.concat([X_train, test], axis=0, ignore_index=True)
data.drop(columns=['Name', 'VIP'], inplace=True)
data.info()
data['Pgroup'] = data['PassengerId'].str.split('_', expand=True)[0].copy()
data['Pgroup'].head()
data['GroupSize'] = 0
groups = list(data.groupby('Pgroup')['Pgroup'].groups.keys())
values = list(data.groupby('Pgroup')['Pgroup'].count())
for (group, value) in zip(groups, values):
    data.loc[data['Pgroup'] == group, 'GroupSize'] = data.loc[data['Pgroup'] == group, 'GroupSize'].apply(lambda x: value)
data.drop(columns=['Pgroup'], inplace=True)
data['GroupSize_M'] = data['GroupSize'].apply(group_class)
data.drop(columns=['GroupSize'], inplace=True)
data.head()
data['DeckSide'] = data['Cabin'].str.split('/', expand=True)[2].copy()
data['CabinNumber'] = data['Cabin'].str.split('/', expand=True)[1].copy()
data['Cabin'] = data['Cabin'].str.split('/', expand=True)[0].copy()
data['Deck_BC'] = data['Cabin'].apply(bool_cabin)
data.drop(columns=['Cabin'], inplace=True)
data.loc[~data['CabinNumber'].isnull(), 'CabinNumber'] = data.loc[~data['CabinNumber'].isnull(), 'CabinNumber'].astype('int32')
labels = ['A', 'B', 'C', 'D', 'E', 'F']
data.loc[~data['CabinNumber'].isnull(), 'CabinGroup'] = pd.cut(data.loc[~data['CabinNumber'].isnull(), 'CabinNumber'], 6, labels=labels)
data['CabinGroup'] = data['CabinGroup'].apply(extract_cabin_group)
data.drop(columns=['CabinNumber'], inplace=True)
data.head()

def encode_DeckSide(value):
    if value == 'P':
        return 1
    elif value == 'S':
        return 0
    else:
        return value

def encode_dest(destination):
    if destination == 'TRAPPIST-1e':
        destination = 1
    elif destination == 'PSO J318.5-22':
        destination = 2
    elif destination == '55 Cancri e':
        destination = 3
    return destination

def encode_planet(planet):
    if planet == 'Earth':
        planet = 1
    elif planet == 'Europa':
        planet = 2
    elif planet == 'Mars':
        planet = 3
    return planet

def encode_bool(value):
    if np.isnan(value):
        return value
    elif value:
        value = 1
    elif ~value:
        value = 0
    return value
data['DeckSide'] = data['DeckSide'].apply(encode_DeckSide)
data['Destination'] = data['Destination'].apply(encode_dest)
data['HomePlanet'] = data['HomePlanet'].apply(encode_planet)
data['CryoSleep'] = data['CryoSleep'].apply(encode_bool)
data.head()
missing = round(100 * data.isnull().sum() / data.shape[0], 2)
missing_df = pd.DataFrame({'% of missing values': missing})
missing_df
numeric_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'DeckSide', 'Deck_BC', 'CabinGroup', 'GroupSize_M', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
knn_imputer = KNNImputer(n_neighbors=2)
data[numeric_columns] = pd.DataFrame(knn_imputer.fit_transform(data[numeric_columns]), columns=numeric_columns)
missing = round(100 * data.isnull().sum() / data.shape[0], 2)
missing_df = pd.DataFrame({'% of missing values': missing})
missing_df

def clean_labels_1(value):
    if value >= 0.5:
        return 1
    else:
        return 0

def clean_labels_2(value):
    if value >= 2.5:
        return 3
    elif value >= 1.5 and value < 2.5:
        return 2
    else:
        return 1
for feature in ['CryoSleep', 'DeckSide', 'Deck_BC', 'CabinGroup', 'GroupSize_M']:
    data[feature] = data[feature].apply(clean_labels_1)
for feature in ['HomePlanet', 'Destination']:
    data[feature] = data[feature].apply(clean_labels_2)
X_train = data[data['PassengerId'].isin(train_id)]
X_train = X_train.drop(columns=['PassengerId'])
X_test = data[data['PassengerId'].isin(test_id)]
X_test = X_test.drop(columns=['PassengerId'])

def plot_learning_curve(X, y, estimator, ax):
    (train_sizes, train_scores, test_scores) = learning_curve(estimator=estimator, X=X, y=y, train_sizes=np.linspace(0.1, 1.0, 10), cv=8, n_jobs=-1)
    train_mean_acc = np.mean(train_scores, axis=1)
    train_std_acc = np.std(train_scores, axis=1)
    test_mean_acc = np.mean(test_scores, axis=1)
    test_std_acc = np.std(test_scores, axis=1)
    f = plt.figure(figsize=(9, 5))
    ax.set_ylim(0.7, 0.95)
    ax.plot(train_sizes, train_mean_acc, label='Training curve', linestyle='-', marker='o', color='g')
    ax.fill_between(train_sizes, train_mean_acc + train_std_acc, train_mean_acc - train_std_acc, alpha=0.15, color='g')
    ax.plot(train_sizes, test_mean_acc, label='Validation curve', linestyle='--', marker='o', color='r')
    ax.fill_between(train_sizes, test_mean_acc + test_std_acc, test_mean_acc - test_std_acc, alpha=0.15, color='r')
    ax.set_xlabel('Training Sample Size')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True, shadow=True)
    return None
xgb_model = xgb.XGBClassifier()
scores = cross_val_score(xgb_model, X_train, y_train, scoring='accuracy', cv=8)
mean_score = np.mean(scores)
std_score = np.std(scores)
print('Best score: {:.1f}% +/- {:.1f}%'.format(mean_score * 100, std_score * 100))