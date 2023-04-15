import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
print(df.info())
df.describe()
print(df.isnull().sum())
df.astype(bool).sum(axis=0)
histogram = df.hist(figsize=(20, 20))
df['Glucose'] = np.where(df['Glucose'] == 0, df['Glucose'].mean(), df['Glucose'])
df['BloodPressure'] = np.where(df['BloodPressure'] == 0, df['BloodPressure'].mean(), df['BloodPressure'])
df['SkinThickness'] = np.where(df['SkinThickness'] == 0, df['SkinThickness'].median(), df['SkinThickness'])
df['Insulin'] = np.where(df['Insulin'] == 0, df['Insulin'].median(), df['Insulin'])
df['BMI'] = np.where(df['BMI'] == 0, df['BMI'].mean(), df['BMI'])
new_histogram = df.hist(figsize=(20, 20))
df.Outcome.value_counts().plot(kind='bar')
plt.figure(figsize=(20, 20))
p = sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
X = df.drop('Outcome', axis=1)
y = df[['Outcome']]
print('Type of X is {} \nType of y is {}'.format(type(X), type(y)))
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier