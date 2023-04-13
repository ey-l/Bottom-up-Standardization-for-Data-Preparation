import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.dtypes
df.info()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print('Column' + '\t\t\t\t Total missing Values' + '\t\t\t\t % of missing values')
for i in df.columns:
    print(f'{i: <50}{df[i].isnull().sum():<30}{df[i].isnull().sum() * 100 / df.shape[0]: .2f}')
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)
print('Column' + '\t\t\t\t Total missing Values' + '\t\t\t\t % of missing values')
for i in df.columns:
    print(f'{i: <50}{df[i].isnull().sum():<30}{df[i].isnull().sum() * 100 / df.shape[0]:.2f}')
pass
pass
pass
pass
X = df.drop('Outcome', axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.preprocessing import StandardScaler
scaling_x = StandardScaler()
X_train = scaling_x.fit_transform(X_train)
X_test = scaling_x.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()