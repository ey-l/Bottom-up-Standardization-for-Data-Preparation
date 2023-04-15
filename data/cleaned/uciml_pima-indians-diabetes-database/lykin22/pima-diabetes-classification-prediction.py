import numpy as np
import pandas as pd
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.shape
df.columns
df.dtypes
df.head()
df.info()
df.describe().T
df.isnull().any()
df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
df.head()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.countplot(x='Outcome', data=df)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Has Diabetes')
plt.ylabel('Count')

df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
df_copy.isnull().sum()
p = df_copy.hist(figsize=(15, 15))
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)
p = df_copy.hist(figsize=(15, 15))
df_copy.isnull().sum()
from sklearn.model_selection import train_test_split
X = df.drop(columns='Outcome')
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def find_best_model(X, y):
    models = {'logistic_regression': {'model': LogisticRegression(solver='lbfgs', multi_class='auto'), 'parameters': {'C': [1, 5, 10]}}, 'decision_tree': {'model': DecisionTreeClassifier(splitter='best'), 'parameters': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10]}}, 'random_forest': {'model': RandomForestClassifier(criterion='gini'), 'parameters': {'n_estimators': [10, 15, 20, 50, 100, 200]}}, 'svm': {'model': SVC(gamma='auto'), 'parameters': {'C': [1, 10, 20], 'kernel': ['rbf', 'linear']}}}
    scores = []
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for (model_name, model_params) in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv=cv_shuffle, return_train_score=False)