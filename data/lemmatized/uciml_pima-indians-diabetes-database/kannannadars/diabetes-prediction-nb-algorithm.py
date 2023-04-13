import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pass
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.dtypes
df.isnull().values.any()
df_hist = df.drop(labels='Outcome', axis=1)
df_hist.hist(stacked=False, bins=100, figsize=(12, 30), layout=(10, 2))
pass
pass
pass
n_true = len(df.loc[df['Outcome'] == 1])
n_false = len(df.loc[df['Outcome'] == 0])
print('Number of true cases: {0} ({1:2.2f}%)'.format(n_true, n_true / (n_true + n_false) * 100))
print('Number of false cases: {0} ({1:2.2f}%)'.format(n_false, n_false / (n_true + n_false) * 100))
from sklearn.model_selection import train_test_split
x = df.drop(labels='Outcome', axis=1)
y = df['Outcome']
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=1)
x_train.head()
print('{0:0.2f}% data is in training set'.format(len(x_train) / len(df.index) * 100))
print('{0:0.2f}% data is in test set'.format(len(x_test) / len(df.index) * 100))
for i in df.columns:
    print(i, (df[i] == 0).sum())
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=0, strategy='mean')
cols = x_train.columns
x_train = pd.DataFrame(imp_mean.fit_transform(x_train))
x_test = pd.DataFrame(imp_mean.fit_transform(x_test))
x_train.columns = cols
x_test.columns = cols
x_train.head(10)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()