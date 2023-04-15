import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
target = train_df.Transported
train_df_all = train_df
train_df = train_df.drop('Transported', axis=1)
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_df.info()

def get_null_row_ratio(df):
    all_row_length = len(df)
    null_row_length = len(df.loc[df.isnull().sum(axis=1).astype(bool)])
    return f'{round(null_row_length / all_row_length * 100, 2)}%'
print('train : ' + get_null_row_ratio(train_df))
print('test : ' + get_null_row_ratio(test_df))

def plot_null_value_count(df1, df2):
    null_values = pd.concat([df1.isna().sum().rename('train'), df2.isna().sum().rename('test')], axis=0).rename('null count').reset_index().rename(columns={'index': 'column'})
    null_values['data'] = ['train'] * len(df1.columns) + ['test'] * len(df2.columns)
    (f, ax) = plt.subplots(figsize=(5, 5))
    sns.barplot(data=null_values, y='column', x='null count', hue='data', orient='h')
plot_null_value_count(train_df, test_df)

def plot_null_value_ratio(df1, df2):
    null_values = pd.concat([df1.isna().sum().rename('train') / len(df1), df2.isna().sum().rename('test') / len(df2)], axis=0).rename('null count').reset_index().rename(columns={'index': 'column'})
    null_values['data'] = ['train'] * len(df1.columns) + ['test'] * len(df2.columns)
    (f, ax) = plt.subplots(figsize=(5, 5))
    sns.barplot(data=null_values, y='column', x='null count', hue='data', orient='h')
plot_null_value_ratio(train_df, test_df)
paid_service_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

def process_null_data(df):
    temp_df = df.copy()
    for column in paid_service_columns:
        temp_df[column] = temp_df[column].fillna(0)
    temp_df['Age'] = temp_df.Age.fillna(-1)
    temp_df['Cabin'] = temp_df.Cabin.fillna('N/-1/N')
    temp_df['HomePlanet'] = temp_df.HomePlanet.fillna('Unknown')
    temp_df['CryoSleep'] = temp_df.CryoSleep.fillna('Unknown')
    temp_df['Destination'] = temp_df.Destination.fillna('Unknown')
    temp_df['VIP'] = temp_df.VIP.fillna('Unknown')
    temp_df['Name'] = temp_df.Name.fillna('N N')
    return temp_df

def split_id(df):
    temp_df = df.copy()
    temp_df['group'] = temp_df.PassengerId.apply(lambda x: x[:4])
    temp_df['Id'] = temp_df.PassengerId.apply(lambda x: x[5:])
    temp_df.drop(['PassengerId'], axis=1, inplace=True)
    return temp_df

def split_name(df):
    temp_df = df.copy()
    temp_df['first_name'] = temp_df.Name.apply(lambda x: str(x).split()[0])
    temp_df['last_name'] = temp_df.Name.apply(lambda x: str(x).split()[1])
    temp_df.drop(['Name'], axis=1, inplace=True)
    return temp_df

def split_cabin(df):
    temp_df = df.copy()
    temp_df['deck'] = temp_df.Cabin.apply(lambda x: str(x).split('/')[0])
    temp_df['num'] = temp_df.Cabin.apply(lambda x: str(x).split('/')[1])
    temp_df['side'] = temp_df.Cabin.apply(lambda x: str(x).split('/')[2])
    temp_df.drop(['Cabin'], axis=1, inplace=True)
    return temp_df

def cat_age(age):
    cat = ''
    if age <= -1:
        cat = 'Unknown'
    elif age <= 5:
        cat = 'Baby'
    elif age <= 12:
        cat = 'Child'
    elif age <= 18:
        cat = 'Teenager'
    elif age <= 25:
        cat = 'Student'
    elif age <= 35:
        cat = 'Young Adult'
    elif age <= 60:
        cat = 'Adult'
    else:
        cat = 'Elderly'
    return cat

def labeling_age(df):
    df['Age_cat'] = df.Age.apply(lambda x: cat_age(x))
    df.drop('Age', axis=1, inplace=True)
    return df

def cat_paid_service(df):
    df['paid_service_cat'] = round(df.loc[:, paid_service_columns].sum(axis=1) / 1000)
    return df

def bool_paid_service(charge):
    if charge <= 0:
        return False
    else:
        return True

def judge_use_paid_service(df):
    for column in paid_service_columns:
        df[column] = df[column].apply(lambda x: bool_paid_service(x))
    return df

def cat_use_paid_service(df):
    df['using_paid_service'] = df.paid_service_cat.apply(lambda x: x != 0)
    return df

def pre_process_df(df):
    return judge_use_paid_service(cat_paid_service(labeling_age(split_cabin(split_name(split_id(process_null_data(df))))))).drop(['num', 'Id', 'first_name'], axis=1)

def val_count_df(df, column_name, sort_by_column_name=False):
    value_count = df[column_name].value_counts().reset_index().rename(columns={column_name: 'count', 'index': column_name}).set_index(column_name)
    value_count = value_count.reset_index()
    if sort_by_column_name:
        value_count = value_count.sort_values(column_name)
    return value_count

def plot_and_display_compare_valuecounts(df1, df2, column_name, sort_by_column_name):
    val_count_1 = val_count_df(df1, column_name, sort_by_column_name)
    val_count_2 = val_count_df(df2, column_name, sort_by_column_name)
    val_count = pd.merge(val_count_1, val_count_2, on=column_name, how='outer')
    val_count = val_count.fillna(0)
    val_count.set_index(column_name).plot.pie(figsize=(12, 7), legend=False, ylabel='', subplots=True, title=['Train: ' + column_name, 'Test: ' + column_name])
train_test = pre_process_df(train_df)
test_test = pre_process_df(test_df)
cat_features = ['HomePlanet', 'Destination', 'deck', 'side', 'Age_cat', 'paid_service_cat']
for column in cat_features:
    plot_and_display_compare_valuecounts(train_test, test_test, column, True)

def bool_to_str(boolean):
    if boolean == True:
        return 'True'
    elif boolean == False:
        return 'False'
    elif boolean == 'Unknown':
        return 'Unknown'

def plot_bool_column_count(df1, df2):
    for column in paid_service_columns + ['CryoSleep', 'VIP']:
        df1[column] = df1[column].apply(lambda x: bool_to_str(x))
        df2[column] = df2[column].apply(lambda x: bool_to_str(x))
        plot_and_display_compare_valuecounts(df1, df2, column, False)
plot_bool_column_count(train_test, test_test)
train_combined = train_test.join(target)
plt.subplots(figsize=(20, 26))
for (i, column) in enumerate(paid_service_columns + cat_features + ['CryoSleep', 'VIP']):
    plt.subplot(5, 3, i + 1)
    sns.barplot(x=column, y='Transported', data=train_combined)