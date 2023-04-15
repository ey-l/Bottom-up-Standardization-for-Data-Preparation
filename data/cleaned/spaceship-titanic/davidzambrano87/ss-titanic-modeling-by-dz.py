import pandas as pd
train_dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_dataset.head()
import numpy as np

def fill_nan(dataset):
    fill_with_mean = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    fill_with_mode = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    fill_with_unkn = ['Cabin', 'Name']
    new_dataset = dataset.copy()
    for var in fill_with_mean:
        new_dataset[var] = new_dataset[var].fillna(new_dataset[var].mean())
    for var in fill_with_mode:
        new_dataset[var] = new_dataset[var].fillna(new_dataset[var].mode().iloc[0])
    for var in fill_with_unkn:
        new_dataset[var] = new_dataset[var].fillna('unkn/un kn/unkn')
    return new_dataset

def passenger_group(dataset):
    new_dataset = dataset.copy()
    new_dataset['PassengerGroup'] = new_dataset['PassengerId'].str.split('_', expand=True)[0].astype('float')
    return new_dataset

def family_size(dataset):
    new_dataset = dataset.copy()
    new_dataset['FamilyName'] = new_dataset['Name'].str.split(' ', expand=True)[1]
    family_name_size = new_dataset['FamilyName'].value_counts().to_dict()
    new_dataset['FamilySize'] = new_dataset['FamilyName'].map(family_name_size)
    return new_dataset

def cabin_deck_side(dataset):
    new_dataset = dataset.copy()
    new_dataset['L_Cabin'] = new_dataset['Cabin'].str.split('/', expand=True)[0]
    new_dataset['Side'] = new_dataset['Cabin'].str.split('/', expand=True)[2]
    return new_dataset

def log_transform(dataset):
    vars_to_transform = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    new_dataset = dataset.copy()
    for var in vars_to_transform:
        new_label = 'log_' + var
        new_dataset[new_label] = np.log(new_dataset[var] + 1)
        new_dataset[new_label] = new_dataset[new_label] / new_dataset[new_label].max()
    return new_dataset

def new_features(dataset):
    new_dataset = dataset.copy()
    new_dataset['exp_1'] = (new_dataset['log_RoomService'] > new_dataset['log_VRDeck']) | (new_dataset['log_RoomService'] > new_dataset['log_Spa'])
    new_dataset['exp_2'] = ~(new_dataset['log_RoomService'] > new_dataset['log_VRDeck']) | ~(new_dataset['log_RoomService'] > new_dataset['log_Spa'])
    new_dataset['exp_3'] = (new_dataset['FoodCourt'] > new_dataset['VRDeck']) | ~(new_dataset['RoomService'] > new_dataset['VRDeck'])
    new_dataset['exp_4'] = (new_dataset['FoodCourt'] > new_dataset['Spa']) | ~(new_dataset['RoomService'] > new_dataset['Spa'])
    for exp_var in new_dataset.columns:
        if exp_var.startswith('exp') == True:
            new_dataset[exp_var] = new_dataset[exp_var].astype(float)
    return new_dataset

def one_hot_encode(dataset):
    new_dataset = dataset.copy()
    variables_to_encode = ['HomePlanet', 'Destination', 'L_Cabin', 'Side', 'FamilySize']
    for var in variables_to_encode:
        cat_labels = list(new_dataset[var].unique())
        dummy_label = var[:4]
        dummy = pd.get_dummies(new_dataset[var], prefix=dummy_label, drop_first=True)
        new_dataset = pd.merge(left=new_dataset, right=dummy, left_index=True, right_index=True)
        for cat in cat_labels:
            column_name = dummy_label + '_' + str(cat)
            try:
                new_dataset[column_name] = new_dataset[column_name].astype(float)
            except:
                continue
    return new_dataset

def remove_variables(dataset):
    new_dataset = dataset.copy()
    variables_to_drop = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet', 'Destination', 'Cabin', 'Name', 'L_Cabin', 'Side', 'FamilyName', 'FamilySize']
    try:
        new_dataset = new_dataset.drop(variables_to_drop, axis=1)
    except:
        new_dataset = new_dataset[variables_to_keep[:-1]]
    return new_dataset

def fix_data_types(dataset):
    new_dataset = dataset.copy()
    vars_to_fix_data_type = ['CryoSleep', 'VIP']
    for var in vars_to_fix_data_type:
        new_dataset[var] = new_dataset[var].astype(float)
    return new_dataset

def apply_fe(dataset):
    new_dataset = dataset.copy()
    new_dataset = fill_nan(new_dataset)
    new_dataset = family_size(new_dataset)
    new_dataset = cabin_deck_side(new_dataset)
    new_dataset = log_transform(new_dataset)
    new_dataset = new_features(new_dataset)
    new_dataset = one_hot_encode(new_dataset)
    new_dataset = remove_variables(new_dataset)
    new_dataset = fix_data_types(new_dataset)
    return new_dataset
train_dataset_clean = train_dataset.dropna()
train_dataset_fe = apply_fe(train_dataset_clean)
train_dataset_fe.head()
import seaborn as sns
import matplotlib.pyplot as plt
fe_vars = ['log_Age', 'log_RoomService', 'log_FoodCourt', 'log_ShoppingMall', 'log_Spa', 'log_VRDeck', 'exp_1', 'exp_2', 'exp_3', 'exp_4', 'VIP', 'CryoSleep', 'Transported']
(fig, axes) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(17, 7))
fig.suptitle('original vs new features correlations')
sns.heatmap(train_dataset.corr(), annot=True, cmap='coolwarm', ax=axes[0], fmt='.2f')
sns.heatmap(train_dataset_fe[fe_vars].corr(), annot=True, cmap='coolwarm', ax=axes[1], fmt='.2f')

from sklearn.model_selection import train_test_split
X = train_dataset_fe.drop('Transported', axis=1)
y = train_dataset_fe['Transported']
(X_train, X_dev, y_train, y_dev) = train_test_split(X, y, test_size=0.2, random_state=1)
(len(X_dev), len(X_train))
X_train.head()
X_dev.head()
y_train = y_train.astype(float)
y_dev = y_dev.astype(float)
test_dataset = pd.read_csv('data/input/spaceship-titanic/test.csv')
test_dataset.head()

def apply_fe(dataset):
    new_dataset = dataset.copy()
    new_dataset = fill_nan(new_dataset)
    new_dataset = family_size(new_dataset)
    new_dataset = cabin_deck_side(new_dataset)
    new_dataset = log_transform(new_dataset)
    new_dataset = new_features(new_dataset)
    new_dataset = one_hot_encode(new_dataset)
    new_dataset = fix_data_types(new_dataset)
    new_dataset = remove_variables(new_dataset)
    return new_dataset
X_test = apply_fe(test_dataset)
X_test.head()
from scipy.stats import ttest_ind

def density_comparison(var1):
    (stat, p) = ttest_ind(X_dev[var1], X_test[var1])
    print('p-value for identical distribution:', p)
    sns.kdeplot(np.log(X_dev[var1] + 1), shade=True, color='r')
    sns.kdeplot(np.log(X_test[var1] + 1), shade=True, color='b')

for var in X_dev.columns:
    try:
        density_comparison(var)
    except:
        continue
import statsmodels.api as sm
logit_model = sm.Logit(y_train, X_train.iloc[:, 1:])