import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
pass
import warnings
warnings.filterwarnings('ignore')
os.getcwd()
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.sample(6)
df.info()
df.isnull().sum()
df.describe().T
df.hist(grid=False, figsize=(12, 16))
pass
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
df.hist(grid=False, figsize=(12, 16))
pass
pass
for cols in df.columns:
    pass
    pass
for col in df.columns:
    pass
    pass
df.corr()['Outcome'].sort_values(ascending=False)
pass
pass
pass
pass
pass
pass
print(df['Outcome'].value_counts())
print('\n\n')
pass
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.25, random_state=101, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
pd.DataFrame(x_train_scaled).describe().T
for col in pd.DataFrame(x_train_scaled):
    pass
    pass
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state=42)