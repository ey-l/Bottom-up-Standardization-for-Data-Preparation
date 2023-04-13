import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1

def get_surname(name):
    return name[name.find(' ') + 1:] if isinstance(name, str) else 'UNKNOWN'

def add_surname_column(dataframe):
    dataframe['Surname'] = dataframe['Name'].map(get_surname)
add_surname_column(_input1)
add_surname_column(_input0)
_input1[['Name', 'Surname']]
cabin_set = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])
cabins_per_deck = dict()

def get_cabin_number(cabin: str) -> int:
    return int(cabin[cabin.find('/') + 1:cabin.rfind('/')])

def get_deck(cabin):
    if isinstance(cabin, str) and cabin[0] in cabin_set:
        deck_number = ord(cabin[0]) - ord('A') if cabin[0] != 'T' else 7
        previous_value = 0 if deck_number not in cabins_per_deck else cabins_per_deck[deck_number]
        cabin_id = get_cabin_number(cabin)
        cabins_per_deck[deck_number] = max(previous_value, cabin_id)
        return deck_number
    return random.randint(0, 7)

def add_deck_column(dataframe):
    dataframe['Deck'] = dataframe['Cabin'].map(get_deck)
add_deck_column(_input1)
add_deck_column(_input0)
print_data = _input1.groupby(['Deck'], sort=False).size().reset_index(name='Count')
print_data['Cabins number'] = print_data['Deck'].map(cabins_per_deck.get)
print_data

def random_cabin_id(row):
    return random.randint(0, cabins_per_deck[row['Deck']])

def add_cabin_num_column(dataframe):
    dataframe['CabinNum'] = dataframe.apply(lambda c: get_cabin_number(c['Cabin']) if isinstance(c['Cabin'], str) else random_cabin_id(c), axis=1)
add_cabin_num_column(_input1)
add_cabin_num_column(_input0)
_input1[['Cabin', 'CabinNum']]
_input1['PassengerId'].map(lambda id: int(id[-2:])).max()

def get_family_group(PassengerId):
    return int(PassengerId[:4])

def add_family_size_column(dataframe):
    dataframe['GroupId'] = dataframe['PassengerId'].map(get_family_group)
    groups_counts = dataframe['GroupId'].value_counts()
    dataframe['FamilySize'] = dataframe.apply(lambda row: groups_counts[row['GroupId']], axis=1)
add_family_size_column(_input1)
add_family_size_column(_input0)
print(f"Max number of family members is : {_input1['FamilySize'].max()}")
_input1[['GroupId', 'Name', 'FamilySize']]

def return_zero_if_nan(value) -> float:
    return 0 if np.isnan(value) else value

def add_more_columns(dataframe):
    dataframe['Fare'] = dataframe['RoomService'].map(return_zero_if_nan) + dataframe['FoodCourt'].map(return_zero_if_nan) + dataframe['ShoppingMall'].map(return_zero_if_nan) + dataframe['Spa'].map(return_zero_if_nan) + dataframe['VRDeck'].map(return_zero_if_nan)
    dataframe['FarePerPerson'] = dataframe['Fare'] / dataframe['FamilySize']
add_more_columns(_input1)
add_more_columns(_input0)
_input1[['Name', 'Fare', 'FarePerPerson']]
count_column = 'Count'
ages = _input1.groupby(['Age'], sort=False).size().reset_index(name=count_column)
transported_ages = _input1.loc[_input1['Transported']]
transported_ages = transported_ages['Age'].value_counts()
total_ages = _input1['Age'].value_counts()
x = ages['Age'].to_numpy()
y = ages[count_column].to_numpy()

def interpolate_color(a: tuple, b: tuple, alpha: float) -> tuple:
    new_color = list(a)
    for (i, color_component) in enumerate(new_color):
        new_color[i] = color_component + alpha * (b[i] - color_component)
    return tuple(new_color)

def color_age(age: float) -> tuple:
    they_alive_color = (1.0, 0.0, 0.0)
    they_transported_color = (0.0, 0.0, 0.0)
    if age not in transported_ages:
        transported_ages[age] = 0
    return interpolate_color(they_alive_color, they_transported_color, transported_ages[age] / total_ages[age])
(fig, ax) = plt.subplots()
ax.bar(x, y, color=[color_age(age) for age in x])
plt.xticks(np.arange(0, max(x), 2.5))
plt.setp(ax.get_xticklabels(), rotation=80, ha='right')
fig.set_figwidth(12)
fig.set_figheight(6)

def fix_nan_age(dataframe):
    dataframe['Age'] = dataframe['Age'].replace(np.nan, dataframe['Age'].median())
fix_nan_age(_input1)
fix_nan_age(_input0)

def get_age_class(age) -> int:
    if isinstance(age, float):
        if 0 <= age < 5:
            return 1
        if 5 <= age < 12.5 or 45 <= age < 65:
            return 2
        if 12.5 <= age < 17.5 or 30 <= age < 45:
            return 3
        if 17.5 <= age < 24:
            return 4
        if 24 <= age < 30:
            return 5
        return 6
    raise ValueError('Bad age passed!')

def add_age_class_column(dataframe):
    dataframe['AgeClass'] = dataframe['Age'].map(get_age_class)
add_age_class_column(_input1)
add_age_class_column(_input0)
_input1[['PassengerId', 'Age', 'AgeClass']]
_input1.dtypes
featuries = ['CryoSleep', 'VIP', 'Deck', 'CabinNum', 'GroupId', 'FamilySize', 'Fare', 'FarePerPerson', 'AgeClass', 'Surname']

def column_to_category(dataframe, column):
    dataframe[column] = dataframe[column].astype('category').cat.codes
for dataframe in [_input1, _input0]:
    for feature in featuries:
        if dataframe.dtypes[feature] == np.dtype(object):
            column_to_category(dataframe, feature)
x_train = _input1[featuries]
y_train = _input1['Transported']
x_test = _input0[featuries]
clf = ExtraTreesClassifier(class_weight='balanced', criterion='gini', max_features='log2', n_estimators=200)
ada = AdaBoostClassifier(base_estimator=clf, n_estimators=200, algorithm='SAMME')