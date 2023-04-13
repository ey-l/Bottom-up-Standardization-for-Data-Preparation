import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
drop_nans = True
fill_nans_with_average = False
fill_nans_with_zeros = True
remove_outliers = False
dummy_optimisation = True
feature_importance_plot = False
variables_to_drop_training = ['PassengerId', 'Num', 'Cabin', 'Group', 'Vowels', 'Consonant']
data = read_csv('data/input/spaceship-titanic/train.csv')
data_orig = read_csv('data/input/spaceship-titanic/train.csv')
data_test = read_csv('data/input/spaceship-titanic/test.csv')
data_sub = read_csv('data/input/spaceship-titanic/test.csv')
nans_fractions = []
print('Looking at the distribution of the NaNs')
for column in data.columns:
    num_of_nans = data[column].isna().sum()
    nans_fractions.append(num_of_nans / len(data))
    print('Column ' + str(column) + ' has ' + str(num_of_nans) + ' nans')

def plot_nans_summary(nans):
    fig = plt.figure()
    ax = sn.barplot(x=data.columns, y=nans, color='lightblue', edgecolor='black')
    ax.set(xlabel='Feature', ylabel='Fraction of NaNs [%]')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('nans_distribution.png')
plot_nans_summary(nans_fractions)

def plot_nans_agg_data_individual(data):
    for column in data.columns:
        fig = plt.figure()
        num_of_nans = data[column].isna().sum()
        if num_of_nans > 0:
            columns_to_plot = data.columns
            data_agg = data.groupby(data[column].isnull(), as_index=False).mean(numeric_only=True)
            data_agg[column] = data_agg.index
            column_means = list(data_agg.mean())[:-1]
            lab_false = list(*np.array(data_agg[data_agg[column] == 0]))[:-1]
            lab_true = list(*np.array(data_agg[data_agg[column] == 1]))[:-1]
            for i in range(0, len(column_means)):
                lab_false[i] = lab_false[i] / column_means[i]
                lab_true[i] = lab_true[i] / column_means[i]
            x_axis = np.arange(len(lab_false))
            plt.bar(x_axis - 0.2 - 0.02, lab_false, 0.4, label='Is not NaN', edgecolor='black', color='lightskyblue')
            plt.bar(x_axis + 0.2 + 0.02, lab_true, 0.4, label='NaN', edgecolor='black', color='plum')
            columns = list(data_agg.columns)[:-1]
            plt.xticks(x_axis, columns)
            plt.xlabel('Features')
            plt.ylabel('A.U.')
            title_plot = 'Average feature value for data with NaNs in ' + column
            plt.title(title_plot)
            plt.legend()
            plt.xticks(rotation=90)
            plt.grid(axis='y')
            plt.tight_layout()
            plot_title = 'agg_data_nans' + column + '.png'
            plt.savefig(plot_title)
plot_nans_agg_data_individual(data)
for column in data.columns:
    print('Analysing column ' + str(column))
    print(data.groupby(data[column].isnull(), as_index=False).mean(numeric_only=True))
from math import nan
data_orig['Deck'] = data_orig['Cabin'].apply(lambda x: x.split('/')[0] if x == x else nan)
data_orig['Num'] = data_orig['Cabin'].apply(lambda x: x.split('/')[1] if x == x else nan)
data_orig['Side'] = data_orig['Cabin'].apply(lambda x: x.split('/')[2] if x == x else nan)
data['Deck'] = data['Cabin'].apply(lambda x: x.split('/')[0] if x == x else nan)
data['Num'] = data['Cabin'].apply(lambda x: int(x.split('/')[1]) if x == x else nan)
data['Side'] = data['Cabin'].apply(lambda x: x.split('/')[2] if x == x else nan)
data_test['Deck'] = data_test['Cabin'].apply(lambda x: x.split('/')[0] if x == x else nan)
data_test['Num'] = data_test['Cabin'].apply(lambda x: int(x.split('/')[1]) if x == x else nan)
data_test['Side'] = data_test['Cabin'].apply(lambda x: x.split('/')[2] if x == x else nan)
data['Group'] = data['PassengerId'].apply(lambda x: int(x.split('_')[0]) if x == x else nan)
data['Group_size'] = data['PassengerId'].apply(lambda x: int(x.split('_')[1]) if x == x else nan)
data_test['Group'] = data_test['PassengerId'].apply(lambda x: int(x.split('_')[0]) if x == x else nan)
data_test['Group_size'] = data_test['PassengerId'].apply(lambda x: int(x.split('_')[1]) if x == x else nan)
data['VIP'] = data['VIP'].map({False: 0, True: 1})
data['CryoSleep'] = data['CryoSleep'].map({False: 0, True: 1})
data['Transported'] = data['Transported'].map({False: 0, True: 1})
data_test['VIP'] = data_test['VIP'].map({False: 0, True: 1})
data_test['CryoSleep'] = data_test['CryoSleep'].map({False: 0, True: 1})
data['Vowels'] = data.Name.str.lower().str.count('[aeiou]')
data['Consonant'] = data.Name.str.lower().str.count('[a-z]') - data['Vowels']
data_orig['Vowels'] = data_orig.Name.str.lower().str.count('[aeiou]')
data_orig['Consonant'] = data_orig.Name.str.lower().str.count('[a-z]') - data_orig['Vowels']
data_test['Vowels'] = data_test.Name.str.lower().str.count('[aeiou]')
data_test['Consonant'] = data_test.Name.str.lower().str.count('[a-z]') - data_test['Vowels']
categorical_data = ['HomePlanet', 'Cabin', 'Destination', 'PassengerId']
for cat_data_to_cnv in categorical_data:
    print('Handling now data category ' + cat_data_to_cnv)
    data[cat_data_to_cnv] = pd.Categorical(data[cat_data_to_cnv]).codes
    data_test[cat_data_to_cnv] = pd.Categorical(data_test[cat_data_to_cnv]).codes
for variable in variables_to_drop_training:
    data = data.drop(variable, axis=1)
    data_test = data_test.drop(variable, axis=1)
if fill_nans_with_zeros:
    for element in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        data[element] = data[element].fillna(0)
        data_test[element] = data_test[element].fillna(0)
        data_orig[element] = data_orig[element].fillna(0)
if drop_nans:
    data = data.dropna()
    data_orig = data_orig.dropna()
if fill_nans_with_average:
    data = data.fillna(data.mean())
    data_orig = data_orig.fillna(data_orig.mean())
data_test = data_test.fillna(data_test.mean())

def detect_outlier(feature):
    outliers = []
    data_feat = data[feature]
    mean = np.mean(data_feat)
    std = np.std(data_feat)
    for y in data_feat:
        z_score = (y - mean) / std
        if np.abs(z_score) > 3:
            outliers.append(y)
    print('\nOutlier caps for {}:'.format(feature))
    print('  --95p: {:.1f} / {} values exceed that'.format(data_feat.quantile(0.95), len([i for i in data_feat if i > data_feat.quantile(0.95)])))
    print('  --3sd: {:.1f} / {} values exceed that'.format(mean + 3 * std, len(outliers)))
    print('  --99p: {:.1f} / {} values exceed that'.format(data_feat.quantile(0.99), len([i for i in data_feat if i > data_feat.quantile(0.99)])))
if remove_outliers:
    for feat in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        detect_outlier(feat)
        data[feat] = data[feat].clip(upper=data[feat].quantile(0.99))
variables_to_drop = ['Name']
for variable in variables_to_drop:
    data = data.drop(variable, axis=1)
    data_test = data_test.drop(variable, axis=1)

def conf_matrix_plot(matrix, plot_name):
    fig = plt.figure(figsize=(10, 10))
    plt.title('Correlation matrix')
    mask = np.triu(matrix)
    ax = sn.heatmap(matrix, annot=True, fmt='.1f', vmin=-1, vmax=1, center=0, cmap='vlag')
    plt.tight_layout()
    plt.savefig('heatmap_' + plot_name + '.png')
conf_matrix_plot(data.corr(), 'correlation_features')
all_cat_data = ['HomePlanet', 'Destination', 'CryoSleep']
for element in all_cat_data:
    print('Unique values for categorical data ' + str(element) + ' =' + str(data[element].nunique()))
simple_cat_data = ['HomePlanet', 'Destination', 'VIP', 'CryoSleep']
for element in simple_cat_data:
    print('\n Looking at categorical data ' + str(element) + '\n')
    print(data_orig[[element, 'Transported']].groupby(element).mean())
categorical_data_cabin = ['Deck', 'Side']
for cat_data_to_cnv in categorical_data_cabin:
    if cat_data_to_cnv in list(data.columns):
        print('Handling now data category ' + cat_data_to_cnv)
        data[cat_data_to_cnv] = pd.Categorical(data[cat_data_to_cnv]).codes
        data_test[cat_data_to_cnv] = pd.Categorical(data_test[cat_data_to_cnv]).codes
decomposed_cabin = ['Deck', 'Side']
for element in decomposed_cabin:
    print('\n Looking at categorical data ' + str(element) + '\n')
    print(data_orig[[element, 'Transported']].groupby(element).mean())
    print('Now looking at the correlation with the target label')
    print(data[element].corr(data['Transported']))
print('\n Looking at Group_size column and average values of the target label\n')
if 'Group_size' in list(data.columns):
    print(data[['Group_size', 'Transported']].groupby('Group_size').mean())
X = data.drop('Transported', axis=1)
y = data['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.4, random_state=42)
(X_val, X_test, y_val, y_test) = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
model = XGBClassifier()
from sklearn.model_selection import GridSearchCV
parameters = {}
if dummy_optimisation:
    parameters = {'max_depth': [5, 6, 7]}
else:
    parameters = {'eta': [0.05 * i for i in range(2, 8)], 'gamma': [0.05 * i for i in range(0, 3)], 'max_depth': [4, 5, 6, 7, 8], 'max_leaves': [0, 1, 2]}
cv = GridSearchCV(model, parameters, cv=5)