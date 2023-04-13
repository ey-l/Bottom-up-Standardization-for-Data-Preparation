import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pima = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima.head()
for column in pima.columns:
    print('{col} has '.format(col=column), pima[pima[column] == 0][column].count(), 'zeros')
df = pima.copy()
features_dealing_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for features in features_dealing_with_zero:
    df[features].replace(0, np.nan, inplace=True)
    df[features].fillna(pima[features].mean(), inplace=True)
df.Insulin = df.Insulin.astype(int)
df.SkinThickness = df.SkinThickness.astype(int)
df.Outcome = df.Outcome.replace(0, -1)
df.Outcome = df.Outcome.astype(int)
df.head()
X = df.iloc[:-100, :-1]
x_test = df.iloc[-100:, :-1]
Y = df.iloc[:-100, -1]
y_test = df.iloc[-100:, -1]
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
kfold = KFold(n_splits=2)
LRclf = LinearRegression()
training_score = []
testing_score = []
for (train_index, test_index) in kfold.split(X):
    (X_train, X_test) = (X.iloc[train_index], X.iloc[test_index])
    (Y_train, Y_test) = (Y.iloc[train_index], Y.iloc[test_index])