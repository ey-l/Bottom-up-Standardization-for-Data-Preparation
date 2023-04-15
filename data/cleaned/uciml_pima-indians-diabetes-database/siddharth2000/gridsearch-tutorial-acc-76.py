import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
dfcopy = df.copy()
dfcopy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dfcopy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
sns.heatmap(dfcopy.isnull(), cmap='Blues')
df.hist(figsize=(20, 20))

dfcopy['Insulin'].fillna(dfcopy['Insulin'].median(), inplace=True)
dfcopy['Glucose'].fillna(dfcopy['Glucose'].mean(), inplace=True)
dfcopy['BMI'].fillna(dfcopy['BMI'].mean(), inplace=True)
dfcopy['BloodPressure'].fillna(dfcopy['BloodPressure'].mean(), inplace=True)
dfcopy['SkinThickness'].fillna(dfcopy['SkinThickness'].median(), inplace=True)
dfcopy.head()
dfcopy.hist(figsize=(20, 20))

sns.heatmap(df.corr(), cmap='Blues')
sns.heatmap(dfcopy.corr(), cmap='Blues')
col = 'Outcome'
sns.scatterplot(x='Insulin', y='SkinThickness', data=df, hue=col)
sns.scatterplot(x='Age', y='Pregnancies', data=df, hue=col)
sns.countplot(df['Outcome'])
sns.pairplot(dfcopy, hue='Outcome')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(dfcopy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = dfcopy.Outcome
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = LogisticRegression()
solvers = ['newton-cg', 'saga', 'liblinear']
penalty = ['l2', 'l1']
c_values = [5, 10, 1.0, 0.1, 0.01, 0.05]
grid = dict(solver=solvers, penalty=penalty, C=c_values)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=3, scoring='accuracy', error_score=0)