from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass
import warnings
warnings.filterwarnings('ignore')
diabetes_data1 = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data1.head()
diabetes_data1.info(verbose=True)
diabetes_data1.describe()
diabetes_data1.describe().T
diabetes_data_copy = diabetes_data1.copy(deep=True)
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_data_copy.isnull().sum())
diabetes_data_copy.isnull().sum()
p = diabetes_data1.hist(figsize=(20, 20))
diabetes_data_copy.isnull().sum()
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)
diabetes_data_copy.isnull().sum()
p = diabetes_data_copy.hist(figsize=(20, 20))
diabetes_data_copy.isnull().sum()
diabetes_data1.shape
pass
pass
pass
pass
pass
pass
pass
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = diabetes_data_copy.Outcome
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)
from sklearn.neighbors import KNeighborsClassifier
test_scores = []
train_scores = []
for i in range(1, 15):
    knn = KNeighborsClassifier(i)