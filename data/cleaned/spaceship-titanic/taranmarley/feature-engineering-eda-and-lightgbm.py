import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
df.dtypes

def detect_NaNs(df_temp):
    print('NaNs in data: ', df_temp.isnull().sum().sum())
    print('******')
    count_nulls = df_temp.isnull().sum().sum()
    if count_nulls > 0:
        for col in df_temp.columns:
            print('NaNs in', col + ': ', df_temp[col].isnull().sum().sum())
    print('******')
    print('')
detect_NaNs(df)
detect_NaNs(test_df)
df['Transported'] = df['Transported'].astype(int)

def detect_duplicates(df_temp):
    print('Duplicates in data: ', df.duplicated().sum())
    return df.duplicated().sum()
detect_duplicates(df)

def seperate_passenger_id(df_temp):
    passenger_class = []
    for (idx, row) in df_temp.iterrows():
        passengerid = str(row['PassengerId'])
        if '_' in passengerid:
            passenger_class.append(int(passengerid.split('_')[1]))
        else:
            passenger_class.append(0)
    df_temp['Passenger Class'] = passenger_class
    return df_temp
df = seperate_passenger_id(df)
test_df = seperate_passenger_id(test_df)

def seperate_cabin(df_temp):
    letters = []
    numbers = []
    final_letters = []
    for (idx, row) in df_temp.iterrows():
        cabin = str(row['Cabin'])
        if '/' in cabin:
            letters.append(cabin.split('/')[0])
            numbers.append(cabin.split('/')[1])
            final_letters.append(cabin.split('/')[2])
        else:
            letters.append(None)
            numbers.append(-1)
            final_letters.append(None)
    df_temp['letters'] = letters
    df_temp['numbers'] = numbers
    df_temp['final_letters'] = final_letters
    return df_temp
df = seperate_cabin(df)
test_df = seperate_cabin(test_df)
df = df.drop(columns='Cabin')
test_df = test_df.drop(columns='Cabin')
df['numbers'] = pd.to_numeric(df['numbers'], errors='ignore')
test_df['numbers'] = pd.to_numeric(test_df['numbers'], errors='ignore')
df.dtypes

import gender_guesser.detector as gender

def predict_gender(df):
    d = gender.Detector()
    gender_predicted = []
    for (idx, row) in df.iterrows():
        name = str(row['Name'])
        if ' ' in name:
            predicted = d.get_gender(name.split(' ')[0])
            if predicted == 'mostly_male':
                predicted = 'male'
            elif predicted == 'mostly_female':
                predicted = 'female'
            gender_predicted.append(predicted)
        else:
            gender_predicted.append('unknown')
    df['gender'] = gender_predicted
    df = pd.get_dummies(df, columns=['gender'])
    return df
df = predict_gender(df)
test_df = predict_gender(test_df)
import gender_guesser.detector as gender

def predict_gender_remove_last_letter(df):
    d = gender.Detector()
    gender_predicted = []
    for (idx, row) in df.iterrows():
        if row['gender'] == 'unknown':
            name = str(row['Name'])
            if ' ' in name:
                predicted = d.get_gender(name.split(' ')[0][:-1])
                if predicted == 'mostly_male':
                    predicted = 'male'
                elif predicted == 'mostly_female':
                    predicted = 'female'
                gender_predicted.append(predicted)
            else:
                gender_predicted.append('unknown')
        else:
            gender_predicted.append(row['gender'])
    df['gender'] = gender_predicted
    df = pd.get_dummies(df, columns=['gender'])
    return df

def last_names(df):
    Last_Names = []
    for (idx, row) in df.iterrows():
        name = str(row['Name'])
        if ' ' in name:
            Last_Names.append(name.split(' ')[-1])
        else:
            Last_Names.append(None)
    df['Name'] = Last_Names
    return df
df = last_names(df)
test_df = last_names(test_df)
df_temp = pd.concat([df.copy(), test_df.copy()], ignore_index=True)
df_temp['Num_Family_Members'] = df_temp.groupby(['Name'])['PassengerId'].transform('nunique')
df['Num_Family_Members'] = df_temp['Num_Family_Members'][:8693].values
test_df['Num_Family_Members'] = df_temp['Num_Family_Members'][8693:].values
df = df.drop(columns=['PassengerId'])
test_df = test_df.drop(columns=['PassengerId'])

def encode_columns(df, columns, test_df=None):
    for col in columns:
        le = preprocessing.LabelEncoder()