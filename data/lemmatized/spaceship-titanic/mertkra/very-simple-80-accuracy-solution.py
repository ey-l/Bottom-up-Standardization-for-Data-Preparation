import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def category_differences(train, test, col):
    train_minus_test = set(train.HomePlanet.unique()) - set(test.HomePlanet.unique())
    test_minus_train = set(test.HomePlanet.unique()) - set(train.HomePlanet.unique())
    print('categories that are in train but not in test for the column {} : {}'.format(col, train_minus_test))
    print('categories that are in test but not in train for the column {} : {}'.format(col, test_minus_train))

def seperate_types(df):
    num_con_cols = []
    num_disc_cols = []
    cat_cols = []
    for (label, value) in df.items():
        if pd.api.types.is_string_dtype(value):
            cat_cols.append(label)
        elif df[label].nunique() > 100:
            num_con_cols.append(label)
        else:
            num_disc_cols.append(label)
    return (num_con_cols, num_disc_cols, cat_cols)

def plot_categories(df, var, target):
    (fig, ax) = plt.subplots(figsize=(8, 4))
    plt.xticks(df.index, df[var], rotation=90)
    ax2 = ax.twinx()
    ax.bar(df.index, df['perc_category'], color='lightgrey')
    ax2.plot(df.index, df[target], color='green', label='Seconds')
    ax.axhline(y=0.05, color='red')
    ax.set_ylabel('percentage of data per category')
    ax.set_xlabel(var)
    ax2.set_ylabel('Average Target per category')

def calculate_mean_target_per_category(df, var, target):
    len_df = len(df)
    temp_df = pd.Series(df[var].value_counts() / len_df).reset_index()
    temp_df.columns = [var, 'perc_category']
    temp_df = temp_df.merge(df.groupby([var])[target].mean().reset_index(), on=var, how='left')
    return temp_df

def unique_count(df, cols):
    for col in cols:
        print(f'{col} has {df[col].nunique()} unique categories')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input2 = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
_input2
_input1 = _input1.drop(['PassengerId', 'Name'], axis=1, inplace=False)
test_passid = _input0.PassengerId
_input0 = _input0.drop(['PassengerId', 'Name'], axis=1, inplace=False)
(len(_input1.columns), len(_input0.columns))
(num_con_cols, num_disc_cols, cat_cols) = seperate_types(_input1)
(num_con_cols, num_disc_cols, cat_cols)
_input1.Cabin
unique_count(_input1, cat_cols)
_input1.Cabin
_input1['Cabin'] = _input1.Cabin.apply(lambda x: str(x)[0] + str(x)[-1])
_input0['Cabin'] = _input0.Cabin.apply(lambda x: str(x)[0] + str(x)[-1])
target = 'Transported'
for col in cat_cols:
    temp = calculate_mean_target_per_category(_input1, col, target)
    plot_categories(temp, col, target)
_input1.loc[_input1.VIP == True, 'VIP'].sum() / len(_input1)
_input0.loc[_input1.VIP == True, 'VIP'].sum() / len(_input0)
_input1 = _input1.drop('VIP', axis=1, inplace=False)
_input0 = _input0.drop('VIP', axis=1, inplace=False)
cat_cols.remove('VIP')
temp = pd.cut(_input1.Age, bins=3, labels=False)
temp_2 = pd.cut(_input0.Age, bins=3, labels=False)
temp
age_dict = {0: 'young', 1: 'middle-aged', 2: 'old'}
temp = temp.map(age_dict)
temp_2 = temp_2.map(age_dict)
temp.head()
temp_2.head()
_input1.Age = temp.copy()
_input0.Age = temp_2.copy()
del temp, temp_2
cat_cols.append('Age')
cat_cols
temp = calculate_mean_target_per_category(_input1, 'Age', 'Transported')
plot_categories(temp, 'Age', 'Transported')
for col in cat_cols:
    category_differences(_input1, _input0, col)
    print('*--------*')
cat_cols
for col in cat_cols:
    _input1[col] = _input1[col].fillna('missing', inplace=False)
    _input0[col] = _input0[col].fillna('missing', inplace=False)
_input1[cat_cols].isna().sum()
_input0[cat_cols].isna().sum()
for col in num_con_cols:
    _input1[col] = _input1[col].fillna(_input1[col].median(), inplace=False)
    _input0[col] = _input0[col].fillna(_input0[col].median(), inplace=False)
_input1.isna().sum()
_input0.isna().sum()
unique_count(_input1, cat_cols)
cat_cols
one_hot_features = cat_cols.copy()
one_hot_features.remove('Cabin')
dummies_train = pd.get_dummies(_input1[one_hot_features], drop_first=True)
dummies_test = pd.get_dummies(_input0[one_hot_features], drop_first=True)
_input1 = _input1.drop(one_hot_features, axis=1, inplace=False)
_input0 = _input0.drop(one_hot_features, axis=1, inplace=False)
_input1 = pd.concat([_input1, dummies_train], axis=1)
_input0 = pd.concat([_input0, dummies_test], axis=1)
(len(_input1.columns), len(_input0.columns))
cabin_dict = _input1.groupby('Cabin')['Transported'].mean()
cabin_dict
cabin_dict = dict(cabin_dict)
_input1.Cabin = _input1.Cabin.map(cabin_dict)
_input0.Cabin = _input0.Cabin.map(cabin_dict)
_input1
_input0
X = _input1.drop('Transported', axis=1)
y = _input1.Transported.astype(int)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()