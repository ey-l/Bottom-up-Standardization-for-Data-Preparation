import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
print('Shape of datasets:', df.shape)
df.isna().sum()
corr = df.corr()
corr
pass
pass
corr['Outcome'].sort_values(ascending=False)
df.describe()
col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in col:
    df[i].replace(0, df[i].mean(), inplace=True)
df.describe()
df.hist(figsize=(20, 15))
df.var()
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = pd.DataFrame(ss.fit_transform(df.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
pass
pass
pass
pass
pass
pass
pass
X = X
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.3, random_state=3)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=1, penalty='l2')