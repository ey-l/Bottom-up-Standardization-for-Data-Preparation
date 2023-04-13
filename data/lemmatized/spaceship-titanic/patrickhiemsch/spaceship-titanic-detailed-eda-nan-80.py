import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def annotate_bar_perc(plot, n_rows, text_size=14, text_pos=(0, 8), prec=2):
    """
    Function that annotates a stacked matplotlib barplot with percentage labels.
    
    """
    conts = plot.containers
    for i in range(len(conts[0])):
        height = sum([cont[i].get_height() for cont in conts])
        text = f'{height / n_rows * 100:.{prec}f}%'
        plot.annotate(text, (conts[0][i].get_x() + conts[0][i].get_width() / 2, height), ha='center', va='center', size=text_size, xytext=text_pos, textcoords='offset points')
    return plot
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.info()
_input1[['CryoSleep', 'VIP']] = _input1[['CryoSleep', 'VIP']].astype('boolean')
_input1.describe(include=float).applymap(lambda x: f'{x:0.2f}')
_input1.describe(include=[object, bool])
_input1.Transported.value_counts().plot(kind='bar', title='Distribution of target variable', xlabel='Transported', rot=0)
sns.pairplot(_input1.drop(['CryoSleep', 'VIP'], axis=1), hue='Transported', palette=['C1', 'C2'], kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.5})
log_train = _input1.copy()
to_log_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
log_train[to_log_cols] = log_train[to_log_cols].apply(lambda col: np.log(col + 1), raw=True, axis=1)
sns.pairplot(log_train.drop(['CryoSleep', 'VIP'], axis=1), hue='Transported', palette=['C1', 'C2'], kind='scatter', diag_kind='kde', plot_kws={'alpha': 0.5})
_input1['TotalExp'] = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(1)
sns.kdeplot(data=_input1, x='Age', hue='Transported', shade=True)
all_exp = _input1[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum().sort_values(ascending=False)
all_exp.name = 'Share of the individual expense categories'
np.round(all_exp / all_exp.sum(), decimals=2)
numeric_features = ['Total Expenses', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
log_train['Total Expenses'] = np.log(_input1.TotalExp + 1)
plt.figure(figsize=(25, 25))
for (i, feat) in enumerate(numeric_features):
    plt.subplot(4, 2, i + 1)
    sns.kdeplot(data=log_train, x=feat, hue='Transported', common_norm=False, shade=True)
    plt.xlabel('log ' + feat, fontsize=12)
    plt.title('Distribution of log ' + feat, fontsize=16)
cat_feat = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
plt.figure(figsize=(25 * 0.75, 15 * 0.75))
for (i, feat) in enumerate(cat_feat):
    plt.subplot(2, 2, i + 1)
    p1 = _input1.groupby(feat)['Transported'].value_counts().unstack().plot(kind='bar', stacked=True, rot=0, ax=plt.gca())
    nrows = _input1.dropna(subset=[feat]).shape[0]
    annotate_bar_perc(p1, nrows)
    y_upper = _input1[feat].value_counts().to_numpy().max() * 1.1
    plt.ylim((0, y_upper))
    plt.title('Distribution of ' + feat, fontsize=18)
    plt.xlabel('')
    plt.xticks(fontsize=16)
plt.subplots_adjust(wspace=0.3)
print(f'Number of rows:                       {len(_input1)}')
print(f'Number of rows with >= 1 NaN-value:   {_input1.isna().any(1).sum()}')
print(f"\nPercentage of 'full'-rows:            {100 - _input1.isna().any(1).sum() / len(_input1) * 100:.2f}%")
_input1.isna().sum().sort_values().plot(kind='barh')
msno.matrix(_input1)
_input1.isna().sum(1).value_counts(normalize=True)[1:]
nan_corr = _input1.drop(['PassengerId', 'Transported'], axis=1)
nan_corr['TotalExp'] = _input1.iloc[:, [7, 8, 9, 10, 11]].sum(axis=1, skipna=False)
nan_corr = nan_corr.drop(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], axis=1)
plt.figure(figsize=(10, 8))
msno.heatmap(nan_corr[nan_corr.isna().any(1)], ax=plt.gca())
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(_input1.corr(), dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(_input1.corr(), vmin=-1, vmax=1, annot=True, cmap=cmap, mask=mask, square=True, linewidths=0.5)
print(f'The train dataset contains {len(_input1)} records of passengers.')
_input1.PassengerId.value_counts().max()
_input1.PassengerId.head(5)

def preprocess_PassengerId(data):
    """
    Preprocess PassengerID. Returns three columns:
        1. GroupID   - Unique ID of the group the passenger is in
        2. GroupPos  - Position in the group, that is assigned to passenger
        3. GroupSize - New feature, that assigns each passenger the size of the group he is part of
    """
    new_ID = data.PassengerId.str.split('_', expand=True)
    new_ID.columns = ['GroupID', 'GroupPos']
    new_ID.GroupPos = new_ID.GroupPos.str.replace('0', '').astype(int)
    group_size_dict = new_ID.groupby('GroupID').max().to_dict()['GroupPos']
    new_ID['GroupSize'] = new_ID.apply(lambda row: group_size_dict[row['GroupID']], axis=1)
    new_ID.GroupID = new_ID.GroupID.str.replace(pat='\\b0+(?=\\d)', repl='', regex=True).astype(int)
    return new_ID
train_old = _input1.copy()
_input1 = pd.concat([train_old, preprocess_PassengerId(train_old)], axis=1)
_input1.head().iloc[:, -3:]
group_size_counts = _input1.groupby('GroupID')['GroupSize'].max().value_counts().reset_index()
group_size_counts.columns = ['GroupSize', 'Counts']
print(f'Total number of groups: {group_size_counts.Counts.sum()}\n')
print(f'Perc. travelling alone: {(_input1.GroupSize == 1).mean() * 100:.2f}%\n')
print(f'Overview of group size and respective count:\n')
print(group_size_counts)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
_input1.groupby('GroupSize')['Transported'].value_counts().unstack().plot(kind='bar', rot=0, title='Transported vs. Group Size', ax=plt.gca())
plt.subplot(1, 2, 2)
_input1.groupby('GroupPos')['Transported'].value_counts().unstack().plot(kind='bar', rot=0, title='Transported vs. Group Position', ax=plt.gca())
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title('GroupSize vs. Age')
sns.boxplot(data=_input1, x='GroupSize', y='Age', ax=plt.gca(), palette='muted')
plt.subplot(1, 2, 2)
plt.title('GroupPos vs. Age')
sns.boxplot(data=_input1, x='GroupPos', y='Age', hue='Transported', ax=plt.gca(), palette='muted')
groupID_transported = _input1[_input1.GroupSize > 1].groupby('GroupID')['Transported'].value_counts().unstack(fill_value=0)
same_fate = (groupID_transported == 0).any(1).sum()
print(f'Percentage of group members in groups>1 sharing same fate: {same_fate / len(groupID_transported) * 100:.2f}%')
home_planet_groups = _input1[_input1.GroupSize > 1].groupby('GroupID')['HomePlanet'].unique().apply(lambda x: len(x))
same_planet = home_planet_groups.value_counts(normalize=True).iloc[0]
print(f'In {same_planet * 100:.2f}% of all groups, each group member comes from the same HomePlanet.')
home_planet_groups = _input1[_input1.GroupSize > 1].groupby('GroupID')['Destination'].unique().apply(lambda x: len(x))
home_planet_groups.value_counts(normalize=True)
mask_group_vip = _input1.groupby('GroupID')['VIP'].transform(lambda x: (len(x) > 1) & (x.sum() >= 1))
_input1[mask_group_vip].groupby('GroupID')['VIP'].sum().value_counts()
sns.violinplot(data=log_train.dropna(subset=['CryoSleep']), x='CryoSleep', y='Total Expenses', hue='Transported')
_input1[_input1['CryoSleep'] == True]['TotalExp'].sum()
cryo_expenses = _input1[['CryoSleep', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].copy()
cryo_expenses['TotalExpNan'] = cryo_expenses[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(1, skipna=False)
mask_nan_geq_1 = cryo_expenses.isna().any(1)
cryo_expenses_nan = cryo_expenses[mask_nan_geq_1]
print(f'Rows with >= 1 NaN (CryoSleep and/or TotalExp):  {mask_nan_geq_1.sum()}')
print(f"All cases with TotalExp=0:                    {cryo_expenses[cryo_expenses.TotalExpNan == 0].dropna(subset=['CryoSleep']).shape[0]}\n")
print(f'All cases with CryoSleep=True & TotalExp=0:   {cryo_expenses[cryo_expenses.CryoSleep & (cryo_expenses.TotalExpNan == 0)].shape[0]}')
print(f'All cases with CryoSleep=False & TotalExp=0:  {cryo_expenses[~cryo_expenses.CryoSleep & (cryo_expenses.TotalExpNan == 0)].shape[0]}')
cryo_expenses['Age'] = _input1.Age.copy()
plt.figure(figsize=(13, 4))
plt.subplot(1, 2, 1)
(bins1, ages1, _) = plt.hist(cryo_expenses[~cryo_expenses.CryoSleep & (cryo_expenses.TotalExpNan == 0)].Age, bins=70)
plt.axvline(12, ymax=0.95, color='C3', ls='--', lw=1, label='Cutoff-Age: 12')
plt.legend()
plt.title('CryoSleep=False | TotalExp=0')
plt.xlabel('Age')
plt.subplot(1, 2, 2)
(bins2, ages2, _) = plt.hist(cryo_expenses[cryo_expenses.CryoSleep & (cryo_expenses.TotalExpNan == 0)].Age, bins=70)
plt.axvline(12, ymax=0.95, color='C3', ls='--', lw=1, label='Cutoff-Age: 12')
plt.legend()
plt.title('CryoSleep=True | TotalExp=0')
plt.xlabel('Age')
plt.suptitle('Age distribution of both imputation cases')
plt.tight_layout()
q1 = (cryo_expenses[~cryo_expenses.CryoSleep & (cryo_expenses.TotalExpNan == 0)].Age <= 12).mean()
q2 = (cryo_expenses[cryo_expenses.CryoSleep & (cryo_expenses.TotalExpNan == 0)].Age > 12).mean()
print(f'CryoSleep=False | TotalExp=0:  {q1 * 100:.2f}% of passengers <= 12')
print(f'CryoSleep=True  | TotalExp=0:  {q2 * 100:.2f}% of passengers  > 12')
decTree = tree.DecisionTreeClassifier(max_depth=1, criterion='entropy')
age_cryo_zero_exp = cryo_expenses[cryo_expenses.TotalExpNan == 0].dropna(subset=['Age', 'CryoSleep'])[['Age', 'CryoSleep']]