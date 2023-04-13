from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe()
df.describe().T
df_new = df.copy()
df_new[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_new[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(df_new.isnull().sum())
p = df.hist(figsize=(10, 10))
df_new['Glucose'].fillna(df_new['Glucose'].mean(), inplace=True)
df_new['BloodPressure'].fillna(df_new['BloodPressure'].mean(), inplace=True)
df_new['SkinThickness'].fillna(df_new['SkinThickness'].median(), inplace=True)
df_new['Insulin'].fillna(df_new['Insulin'].median(), inplace=True)
df_new['BMI'].fillna(df_new['BMI'].mean(), inplace=True)
p = df_new.hist(figsize=(10, 10))
df.shape
pass
pass
pass
import missingno as msno
pmis = msno.bar(df_new)
p = df.Outcome.value_counts().plot(kind='bar')
from pandas.plotting import scatter_matrix
p = scatter_matrix(df, figsize=(20, 20))
pass
pass
pass
pass
pass
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(df_new.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = df_new.Outcome
y.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_score = []
train_score = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)