import pandas as pd
train_dataset = pd.read_csv('data/input/spaceship-titanic/train.csv')
train_dataset.head()
import matplotlib.pyplot as plt
import missingno as msnum
msnum.matrix(train_dataset)


def fill_nan(dataset):
    fill_with_mean = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    fill_with_mode = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
    new_dataset = dataset
    for var in fill_with_mean:
        new_dataset[var] = new_dataset[var].fillna(new_dataset[var].mean())
    for var in fill_with_mode:
        new_dataset[var] = new_dataset[var].fillna(new_dataset[var].mode().iloc[0])
    return new_dataset
train_dataset_fe = fill_nan(train_dataset)
import numpy as np

def log_transform(dataset):
    vars_to_transform = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    new_dataset = dataset.copy()
    for var in vars_to_transform:
        new_label = 'log_' + var
        new_dataset[new_label] = np.log(new_dataset[var] + 1)
        new_dataset[new_label] = new_dataset[new_label] / new_dataset[new_label].max()
    return new_dataset
train_dataset_fe = log_transform(train_dataset)
import seaborn as sns

def density_comparison_fe(var1):
    (fig, axes) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15, 4))
    title = var1 + ' original vs log_transformed distribution'
    fig.suptitle(title)
    sns.kdeplot(train_dataset_fe[var1], shade=True, color='r', ax=axes[0])
    sns.kdeplot(train_dataset_fe['log_' + var1], shade=True, color='g', ax=axes[1])

vars_to_transform = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for var in vars_to_transform:
    density_comparison_fe(var)
log_columns = ['log_Age', 'log_RoomService', 'log_FoodCourt', 'log_ShoppingMall', 'log_Spa', 'log_VRDeck', 'Transported']
(fig, axes) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15, 6))
fig.suptitle('original vs log_transformed correlations')
sns.heatmap(train_dataset.corr(), annot=True, cmap='coolwarm', ax=axes[0])
sns.heatmap(train_dataset_fe[log_columns].corr(), annot=True, cmap='coolwarm', ax=axes[1])

sns.pairplot(train_dataset_fe[log_columns], hue='Transported')

(fig, axes) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(18, 3))
temp_df = train_dataset[['Transported', 'RoomService', 'VRDeck', 'Spa']].copy()
temp_df['dummy_expenses_TT'] = (temp_df['RoomService'] > temp_df['VRDeck']) | (temp_df['RoomService'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_TT'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[0])
temp_df['dummy_expenses_TF'] = (temp_df['RoomService'] > temp_df['VRDeck']) | ~(temp_df['RoomService'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_TF'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[1])
temp_df['dummy_expenses_FT'] = ~(temp_df['RoomService'] > temp_df['VRDeck']) | (temp_df['RoomService'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_FT'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[2])
temp_df['dummy_expenses_FF'] = ~(temp_df['RoomService'] > temp_df['VRDeck']) | ~(temp_df['RoomService'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_FF'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[3])

sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm')

(fig, axes) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(18, 3))
temp_df = train_dataset[['Transported', 'FoodCourt', 'VRDeck', 'Spa']].copy()
temp_df['dummy_expenses_TT'] = (temp_df['FoodCourt'] > temp_df['VRDeck']) | (temp_df['FoodCourt'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_TT'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[0])
temp_df['dummy_expenses_TF'] = (temp_df['FoodCourt'] > temp_df['VRDeck']) | ~(temp_df['FoodCourt'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_TF'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[1])
temp_df['dummy_expenses_FT'] = ~(temp_df['FoodCourt'] > temp_df['VRDeck']) | (temp_df['FoodCourt'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_FT'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[2])
temp_df['dummy_expenses_FF'] = ~(temp_df['FoodCourt'] > temp_df['VRDeck']) | ~(temp_df['FoodCourt'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_FF'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[3])

sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm')

(fig, axes) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(18, 3))
temp_df = train_dataset[['Transported', 'FoodCourt', 'RoomService', 'VRDeck', 'Spa']].copy()
temp_df['dummy_expenses_TT'] = (temp_df['FoodCourt'] > temp_df['VRDeck']) | ~(temp_df['RoomService'] > temp_df['VRDeck'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_TT'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[0])
temp_df['dummy_expenses_TF'] = (temp_df['FoodCourt'] > temp_df['Spa']) | ~(temp_df['RoomService'] > temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_TF'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[1])
temp_df['dummy_expenses_FT'] = ~(temp_df['FoodCourt'] > temp_df['VRDeck']) & ~(temp_df['RoomService'] < temp_df['VRDeck'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_FT'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[2])
temp_df['dummy_expenses_FF'] = ~(temp_df['FoodCourt'] > temp_df['Spa']) & ~(temp_df['RoomService'] < temp_df['Spa'])
sns.heatmap(pd.crosstab(temp_df['dummy_expenses_FF'], temp_df['Transported']), annot=True, fmt='.0f', cmap='coolwarm', ax=axes[3])

sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm')

train_dataset_fe['exp_1'] = (train_dataset_fe['RoomService'] > train_dataset_fe['VRDeck']) | (train_dataset_fe['RoomService'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_2'] = (train_dataset_fe['RoomService'] > train_dataset_fe['VRDeck']) | ~(train_dataset_fe['RoomService'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_3'] = ~(train_dataset_fe['RoomService'] > train_dataset_fe['VRDeck']) | (train_dataset_fe['RoomService'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_4'] = ~(train_dataset_fe['RoomService'] > train_dataset_fe['VRDeck']) | ~(train_dataset_fe['RoomService'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_5'] = (train_dataset_fe['FoodCourt'] > train_dataset_fe['VRDeck']) | (train_dataset_fe['FoodCourt'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_6'] = (train_dataset_fe['FoodCourt'] > train_dataset_fe['VRDeck']) | ~(train_dataset_fe['FoodCourt'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_7'] = ~(train_dataset_fe['FoodCourt'] > train_dataset_fe['VRDeck']) | (train_dataset_fe['FoodCourt'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_8'] = ~(train_dataset_fe['FoodCourt'] > train_dataset_fe['VRDeck']) | ~(train_dataset_fe['FoodCourt'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_9'] = (train_dataset_fe['FoodCourt'] > train_dataset_fe['VRDeck']) | ~(train_dataset_fe['RoomService'] > train_dataset_fe['VRDeck'])
train_dataset_fe['exp_10'] = (train_dataset_fe['FoodCourt'] > train_dataset_fe['Spa']) | ~(train_dataset_fe['RoomService'] > train_dataset_fe['Spa'])
train_dataset_fe['exp_11'] = ~(train_dataset_fe['FoodCourt'] > train_dataset_fe['VRDeck']) & ~(train_dataset_fe['RoomService'] < train_dataset_fe['VRDeck'])
train_dataset_fe['exp_12'] = ~(train_dataset_fe['FoodCourt'] > train_dataset_fe['Spa']) & ~(train_dataset_fe['RoomService'] > train_dataset_fe['Spa'])
fe_vars = ['Transported', 'CryoSleep', 'VIP', 'log_Age', 'log_RoomService', 'log_FoodCourt', 'log_ShoppingMall', 'log_Spa', 'log_VRDeck', 'exp_1', 'exp_2', 'exp_3', 'exp_4', 'exp_5', 'exp_6', 'exp_7', 'exp_8', 'exp_9', 'exp_10', 'exp_11', 'exp_12']
plt.subplots(figsize=(15, 9))
sns.heatmap(train_dataset_fe[fe_vars].corr(), annot=True, cmap='coolwarm', fmt='.2f')

home_dummy = pd.get_dummies(train_dataset_fe['HomePlanet'], prefix='Home')
train_dataset_fe = pd.merge(left=train_dataset_fe, right=home_dummy, left_index=True, right_index=True)
destination_dummy = pd.get_dummies(train_dataset_fe['Destination'], prefix='Dest')
train_dataset_fe = pd.merge(left=train_dataset_fe, right=destination_dummy, left_index=True, right_index=True)
temp_df = train_dataset_fe[['Transported', 'Home_Earth', 'Home_Europa', 'Home_Mars', 'Dest_55 Cancri e', 'Dest_PSO J318.5-22', 'Dest_TRAPPIST-1e']]
plt.subplots(figsize=(12, 9))
sns.heatmap(temp_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
