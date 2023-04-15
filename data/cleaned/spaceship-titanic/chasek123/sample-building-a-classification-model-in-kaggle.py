import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
path = 'data/input/spaceship-titanic/'
training_dataset = 'train.csv'
titanic_train = pd.read_csv(filepath_or_buffer=path + training_dataset)
titanic_train.info(verbose=True)
path = 'data/input/spaceship-titanic/'
testing_dataset = 'test.csv'
titanic_test = pd.read_csv(filepath_or_buffer=path + testing_dataset)
titanic_test.info(verbose=True)
titanic_train['set'] = 'Training'
titanic_test['set'] = 'Testing'
titanic_df = titanic_train.append(other=titanic_test)
titanic_df.reset_index(drop=False, inplace=True)
titanic_corr = titanic_train.corr(method='pearson').round(decimals=2)
titanic_corr.loc[:, 'Transported'].apply(func=abs).sort_values(ascending=False)
titanic_df['RoomService'].fillna(value=0, inplace=True)
x_train = titanic_df[['RoomService']][titanic_df['set'] == 'Training']
y_train = titanic_df['Transported'][titanic_df['set'] == 'Training']
(x_train_1, x_train_2, y_train_1, y_train_2) = train_test_split(x_train, y_train.astype(dtype='int'), random_state=123, test_size=0.25, stratify=y_train)
model_name = 'Logistic Regression'
model = sklearn.linear_model.LogisticRegression()