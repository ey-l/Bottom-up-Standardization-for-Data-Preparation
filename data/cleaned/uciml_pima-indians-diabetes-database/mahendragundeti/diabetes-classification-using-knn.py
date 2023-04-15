import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
Diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
Diabetes.head()
Diabetes.info(verbose=True)
Diabetes.describe().T
Diabetes_copy = Diabetes.copy(deep=True)
Diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = Diabetes_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
Diabetes_copy.isnull().sum()
Diabetes_copy.hist(figsize=(20, 20))
plt.suptitle('Histogram distribution of our Dataset Features', ha='center', va='top', size=26, fontweight='bold', color='#00008B')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
Diabetes_copy['Glucose'].fillna(Diabetes_copy['Glucose'].mean(), inplace=True)
Diabetes_copy['BloodPressure'].fillna(Diabetes_copy['BloodPressure'].mean(), inplace=True)
Diabetes_copy['SkinThickness'].fillna(Diabetes_copy['SkinThickness'].median(), inplace=True)
Diabetes_copy['Insulin'].fillna(Diabetes_copy['Insulin'].median(), inplace=True)
Diabetes_copy['BMI'].fillna(Diabetes_copy['BMI'].median(), inplace=True)
Diabetes_copy.isnull().sum()
Diabetes_copy.hist(figsize=(20, 20))
plt.suptitle('Histogram distribution of  Features after imputation with Mean and Median', ha='center', va='top', size=26, fontweight='bold', color='#00008B')

sns.countplot(y=Diabetes_copy.dtypes, data=Diabetes_copy)
plt.xlabel('Count of datatypes', fontweight='bold')
plt.ylabel('Datatypes', fontweight='bold')

import missingno as msno
p = msno.bar(Diabetes)
print(Diabetes_copy['Outcome'].value_counts())
Diabetes_copy['Outcome'].value_counts().plot(kind='bar')
plt.xlabel('Is Diabetic or not', fontweight='bold')
plt.ylabel('Count', fontweight='bold')

sns.pairplot(Diabetes_copy, hue='Outcome')
plt.figure(figsize=(14, 12))
sns.heatmap(Diabetes_copy.corr(), annot=True, cmap='RdYlGn', linewidths=2, linecolor='white')
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
y = Diabetes_copy['Outcome']
x = Diabetes_copy.drop(['Outcome'], axis=1)
Diabetes_copy.columns
kk = ss.fit_transform(x)
X = pd.DataFrame(kk, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 4, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 110):
    knn = KNeighborsClassifier(i)