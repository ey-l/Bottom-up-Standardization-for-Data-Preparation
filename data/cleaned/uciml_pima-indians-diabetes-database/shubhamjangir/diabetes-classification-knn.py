from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes.head(5)
diabetes.info(verbose=True)
diabetes.describe()
diabetes.describe().T
diabetes_copy = diabetes.copy(deep=True)
diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_copy.isnull().sum())
a = diabetes.hist(figsize=(20, 20))
diabetes_copy['Glucose'].fillna(diabetes_copy['Glucose'].mean(), inplace=True)
diabetes_copy['BloodPressure'].fillna(diabetes_copy['BloodPressure'].mean(), inplace=True)
diabetes_copy['SkinThickness'].fillna(diabetes_copy['SkinThickness'].median(), inplace=True)
diabetes_copy['Insulin'].fillna(diabetes_copy['Insulin'].median(), inplace=True)
diabetes_copy['BMI'].fillna(diabetes_copy['BMI'].median(), inplace=True)
p = diabetes_copy.hist(figsize=(20, 20))
diabetes.shape
diabetes.dtypes.value_counts()
print(diabetes.dtypes)
import missingno as msno
a = msno.bar(diabetes)
color_wheel = {1: '#0392cf', 2: '#7bc043'}
colors = diabetes['Outcome'].map(lambda x: color_wheel.get(x + 1))
print(diabetes.Outcome.value_counts())
a = diabetes.Outcome.value_counts().plot(kind='bar')
from pandas.plotting import scatter_matrix
p = scatter_matrix(diabetes, figsize=(25, 25))
a = sns.pairplot(diabetes_copy, hue='Outcome')
plt.figure(figsize=(12, 10))
p = sns.heatmap(diabetes.corr(), annot=True, cmap='RdYlGn')
plt.figure(figsize=(12, 10))
p = sns.heatmap(diabetes_copy.corr(), annot=True, cmap='RdYlGn')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head(5)
y = diabetes_copy.Outcome
y.head(5)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)