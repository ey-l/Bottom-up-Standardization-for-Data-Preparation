import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df
diabetes_df.info()
diabetes_df.describe()
diabetes_df.isna().sum()
import seaborn as sns
corr = diabetes_df.corr()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pd.plotting.scatter_matrix(diabetes_df, alpha=0.2, figsize=(15, 10))
diabetes_df['Outcome'].value_counts().plot(kind='bar')
diabetes_df['Insulin'][diabetes_df['Insulin'] == 0].value_counts()
for i in diabetes_df.columns:
    if i != 'Outcome':
        print(diabetes_df[i][diabetes_df[i] == 0].value_counts(), '\n----------------------\n')
X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()