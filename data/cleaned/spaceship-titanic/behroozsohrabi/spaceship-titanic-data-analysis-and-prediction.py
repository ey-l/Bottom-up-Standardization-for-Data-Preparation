import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
print('Train Dataset Shape:', train_df.shape)
print('Test Dataset Shape:', test_df.shape)
train_df.head(5)
df = pd.concat([train_df, test_df])
df.reset_index(drop=True, inplace=True)
print('# of Null values in Transported column:\t\t\t', df['Transported'].isnull().sum(), '\nMust be equal to # of rows in the test dataframe:\t', test_df.shape[0])
df['Destination'].unique()
df.replace({'Destination': {'TRAPPIST-1e': 'A', 'PSO J318.5-22': 'B', '55 Cancri e': 'C'}}, inplace=True)
df[['PassengerGroup', 'PassengerNumber']] = df['PassengerId'].apply(lambda x: pd.Series(str(x).split('_')))
df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].apply(lambda x: pd.Series(str(x).split('/')))
df[['PassengerGroup', 'PassengerNumber', 'CabinNum']] = df[['PassengerGroup', 'PassengerNumber', 'CabinNum']].apply(pd.to_numeric)
df.replace(['None', 'nan'], np.nan, inplace=True)
numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'PassengerGroup', 'PassengerNumber', 'CabinNum']
categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported', 'CabinDeck', 'CabinSide']
df.loc[:, numerical_columns].hist(figsize=(15, 10), rwidth=0.8)
pass
(fig, axes) = plt.subplots(3, 3, figsize=(15, 10))
for (i, column) in enumerate(categorical_columns):
    ax = axes[i // 3, i % 3]
    df[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(column)
    ax.tick_params(axis='x', rotation=0)

df = df.drop(columns=['Name', 'PassengerId', 'Cabin'])
df['Nulls'] = df[df.columns.symmetric_difference(['Transported'])].isnull().sum(axis=1)
df[df.columns.symmetric_difference(['Transported', 'Nulls'])].isnull().sum().sort_values(ascending=False).plot(kind='bar', figsize=(15, 5), title='Null values in each column')
pass
null_values = df[df['Nulls'] > 0]['Nulls']
null_values.agg(['min', 'mean', 'max', 'count'])
hist = null_values.hist(bins=np.arange(1, 7) - 0.5, rwidth=0.8)
plt.title('Distribution of null values in training dataset')
plt.xticks(range(1, 7))
for (i, v) in null_values.value_counts().items():
    plt.annotate(v, (i, v + 20), ha='center')
pass

def cross_chart(df, features, categorical_columns):
    unique_values = {}
    columns = set(df.columns) - set(features)
    for feature in features:
        unique_values[feature] = df.value_counts(feature).keys()
    combinations = it.product(*unique_values.values())
    (cols, rows) = (len(columns), len(list(it.product(*unique_values.values()))))
    (fig, axes) = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    for (i, combination) in enumerate(combinations):
        query = []
        for (j, feature) in enumerate(features):
            if type(combination[j]) == str:
                query.append(f'{feature} == "{combination[j]}"')
            else:
                query.append(f'{feature} == {combination[j]}')
        query = ' & '.join(query)
        for (k, column) in enumerate(columns):
            ax = axes[i, k]
            ax.set_title(column + '\n' + query.replace(' == ', ':').replace('"', ''), fontsize=8)
            df_sub = df.loc[df.query(query).index, column]
            if len(df_sub) > 0:
                if column in categorical_columns:
                    df_sub.value_counts().plot(kind='bar', ax=ax)
                    ax.tick_params(axis='x', rotation=0)
                else:
                    df_sub.hist(ax=ax, rwidth=0.8)
cross_chart(df, ['HomePlanet'], categorical_columns)
before = len(df[df['HomePlanet'].isnull()])
df.loc[df.query('HomePlanet != HomePlanet & Destination == "B"').index, 'HomePlanet'] = 'Earth'
df.loc[df.query('HomePlanet != HomePlanet & (CabinDeck == "A" | CabinDeck == "B" | CabinDeck == "C" | CabinDeck == "T")').index, 'HomePlanet'] = 'Europa'
df.loc[df.query('HomePlanet != HomePlanet & CabinDeck == "G"').index, 'HomePlanet'] = 'Earth'
df.loc[df.query('HomePlanet != HomePlanet & CabinDeck == "F" & VIP == True').index, 'HomePlanet'] = 'Mars'
after = len(df[df['HomePlanet'].isnull()])
print(f'{before - after} out of {before} Null values were filled.\nNull values on HomePlanet:')
print(df[df['HomePlanet'].isnull()].value_counts('CabinDeck'))
cross_chart(df[(df['CabinDeck'] == 'D') | (df['CabinDeck'] == 'E') | (df['CabinDeck'] == 'F')], ['CabinDeck', 'HomePlanet'], categorical_columns)
before = len(df[df['HomePlanet'].isnull()])
df.loc[df.query('HomePlanet != HomePlanet & CabinDeck == "D" & RoomService > 0').index, 'HomePlanet'] = 'Mars'
df.loc[df.query('HomePlanet != HomePlanet & CabinDeck == "D" & FoodCourt > 2000').index, 'HomePlanet'] = 'Europa'
df.loc[df.query('HomePlanet != HomePlanet & CabinDeck == "D" & VRDeck > 1000').index, 'HomePlanet'] = 'Europa'
df.loc[df.query('HomePlanet != HomePlanet & CabinDeck == "E" & VRDeck > 3000').index, 'HomePlanet'] = 'Europa'
df.loc[df.query('HomePlanet != HomePlanet & CabinDeck == "E" & FoodCourt > 3000').index, 'HomePlanet'] = 'Europa'
after = len(df[df['HomePlanet'].isnull()])
print(f'{before - after} out of {before} Null values were filled.')
df['Amenities'] = df.apply(lambda row: sum([row[column] for column in ['FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]), axis=1)
cross_chart(df, ['CryoSleep'], categorical_columns)
before = len(df[df['CryoSleep'].isnull()])
df.loc[df.query('CryoSleep != CryoSleep & (RoomService > 0 | Amenities > 0)').index, 'CryoSleep'] = False
after = len(df[df['CryoSleep'].isnull()])
print(f'{before - after} out of {before} Null values were filled.')