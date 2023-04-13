import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().values.any()
df.info()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
corrmat = df.corr()
top_corr_features = corrmat.index
pass
pass
corrmat
df.head()
diabetes_count = len(df.loc[df['Outcome'] == 1])
not_diabetes_count = len(df.loc[df['Outcome'] == 0])
(diabetes_count, not_diabetes_count)
from sklearn.model_selection import train_test_split
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42)
for feature in top_corr_features:
    print(f'Number of entries are missing are {len(df.loc[df[feature] == 0])}')
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0, strategy='mean')
x_test = imp.fit_transform(x_test)
x_train = imp.fit_transform(x_train)
from sklearn.ensemble import RandomForestClassifier
r_f_model = RandomForestClassifier(random_state=42)