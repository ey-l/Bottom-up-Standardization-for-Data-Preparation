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
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
sample_submission = pd.read_csv('data/input/spaceship-titanic/sample_submission.csv')
sample_submission
train.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test_passid = test.PassengerId
test.drop(['PassengerId', 'Name'], axis=1, inplace=True)
(len(train.columns), len(test.columns))
(num_con_cols, num_disc_cols, cat_cols) = seperate_types(train)
(num_con_cols, num_disc_cols, cat_cols)
train.Cabin
unique_count(train, cat_cols)
train.Cabin
train['Cabin'] = train.Cabin.apply(lambda x: str(x)[0] + str(x)[-1])
test['Cabin'] = test.Cabin.apply(lambda x: str(x)[0] + str(x)[-1])
target = 'Transported'
for col in cat_cols:
    temp = calculate_mean_target_per_category(train, col, target)
    plot_categories(temp, col, target)
train.loc[train.VIP == True, 'VIP'].sum() / len(train)
test.loc[train.VIP == True, 'VIP'].sum() / len(test)
train.drop('VIP', axis=1, inplace=True)
test.drop('VIP', axis=1, inplace=True)
cat_cols.remove('VIP')
temp = pd.cut(train.Age, bins=3, labels=False)
temp_2 = pd.cut(test.Age, bins=3, labels=False)
temp
age_dict = {0: 'young', 1: 'middle-aged', 2: 'old'}
temp = temp.map(age_dict)
temp_2 = temp_2.map(age_dict)
temp.head()
temp_2.head()
train.Age = temp.copy()
test.Age = temp_2.copy()
del temp, temp_2
cat_cols.append('Age')
cat_cols
temp = calculate_mean_target_per_category(train, 'Age', 'Transported')
plot_categories(temp, 'Age', 'Transported')
for col in cat_cols:
    category_differences(train, test, col)
    print('*--------*')
cat_cols
for col in cat_cols:
    train[col].fillna('missing', inplace=True)
    test[col].fillna('missing', inplace=True)
train[cat_cols].isna().sum()
test[cat_cols].isna().sum()
for col in num_con_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)
train.isna().sum()
test.isna().sum()
unique_count(train, cat_cols)
cat_cols
one_hot_features = cat_cols.copy()
one_hot_features.remove('Cabin')
dummies_train = pd.get_dummies(train[one_hot_features], drop_first=True)
dummies_test = pd.get_dummies(test[one_hot_features], drop_first=True)
train.drop(one_hot_features, axis=1, inplace=True)
test.drop(one_hot_features, axis=1, inplace=True)
train = pd.concat([train, dummies_train], axis=1)
test = pd.concat([test, dummies_test], axis=1)
(len(train.columns), len(test.columns))
cabin_dict = train.groupby('Cabin')['Transported'].mean()
cabin_dict
cabin_dict = dict(cabin_dict)
train.Cabin = train.Cabin.map(cabin_dict)
test.Cabin = test.Cabin.map(cabin_dict)
train
test
X = train.drop('Transported', axis=1)
y = train.Transported.astype(int)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()