import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.dtypes

def detect_NaNs(df_temp):
    print('NaNs in data: ', df_temp.isnull().sum().sum())
    print('******')
    count_nulls = df_temp.isnull().sum().sum()
    if count_nulls > 0:
        for col in df_temp.columns:
            print('NaNs in', col + ': ', df_temp[col].isnull().sum().sum())
    print('******')
    print('')
detect_NaNs(_input1)
detect_NaNs(_input0)
_input1['Transported'] = _input1['Transported'].astype(int)

def detect_duplicates(df_temp):
    print('Duplicates in data: ', _input1.duplicated().sum())
    return _input1.duplicated().sum()
detect_duplicates(_input1)

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
_input1 = seperate_passenger_id(_input1)
_input0 = seperate_passenger_id(_input0)

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
_input1 = seperate_cabin(_input1)
_input0 = seperate_cabin(_input0)
_input1 = _input1.drop(columns='Cabin')
_input0 = _input0.drop(columns='Cabin')
_input1['numbers'] = pd.to_numeric(_input1['numbers'], errors='ignore')
_input0['numbers'] = pd.to_numeric(_input0['numbers'], errors='ignore')
_input1.dtypes
import gender_guesser.detector as gender

def predict_gender(df):
    d = gender.Detector()
    gender_predicted = []
    for (idx, row) in _input1.iterrows():
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
    _input1['gender'] = gender_predicted
    _input1 = pd.get_dummies(_input1, columns=['gender'])
    return _input1
_input1 = predict_gender(_input1)
_input0 = predict_gender(_input0)
import gender_guesser.detector as gender

def predict_gender_remove_last_letter(df):
    d = gender.Detector()
    gender_predicted = []
    for (idx, row) in _input1.iterrows():
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
    _input1['gender'] = gender_predicted
    _input1 = pd.get_dummies(_input1, columns=['gender'])
    return _input1

def last_names(df):
    Last_Names = []
    for (idx, row) in _input1.iterrows():
        name = str(row['Name'])
        if ' ' in name:
            Last_Names.append(name.split(' ')[-1])
        else:
            Last_Names.append(None)
    _input1['Name'] = Last_Names
    return _input1
_input1 = last_names(_input1)
_input0 = last_names(_input0)
df_temp = pd.concat([_input1.copy(), _input0.copy()], ignore_index=True)
df_temp['Num_Family_Members'] = df_temp.groupby(['Name'])['PassengerId'].transform('nunique')
_input1['Num_Family_Members'] = df_temp['Num_Family_Members'][:8693].values
_input0['Num_Family_Members'] = df_temp['Num_Family_Members'][8693:].values
_input1 = _input1.drop(columns=['PassengerId'])
_input0 = _input0.drop(columns=['PassengerId'])

def encode_columns(df, columns, test_df=None):
    for col in columns:
        le = preprocessing.LabelEncoder()