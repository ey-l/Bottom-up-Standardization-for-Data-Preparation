from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
pass
import warnings
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
warnings.filterwarnings('ignore')
diabetes_data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_data.head()
diabetes_data.describe()
diabetes_data_copy = diabetes_data.copy(deep=True)
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_data_copy.isnull().sum())
diabetes_data.hist(figsize=(10, 10))
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)
diabetes_data_copy.isna().sum()
paftercleaning = diabetes_data_copy.hist(figsize=(10, 10))
df = pd.DataFrame(diabetes_data)
dfupdated = pd.DataFrame(diabetes_data_copy)
print(diabetes_data.shape, diabetes_data_copy.shape)
pass
pass
pass
diabetes_data.info(verbose=True)
p = msno.bar(diabetes_data)
diabetes_data.Outcome.value_counts().plot(kind='pie')
print('The below graph shows that the data is biased towards datapoints having outcome value as 0 where it means that diabetes was not present actually. \nThe number of non-diabetics is almost twice the number of diabetic patients\n', diabetes_data.Outcome.value_counts())
p = scatter_matrix(diabetes_data, figsize=(25, 25))
pass
pass
pass
pass
pass
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
y = diabetes_data_copy.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=40)
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
models = []
models.append(('Logistic Regression              ', LogisticRegression()))
models.append(('Linear Discriminant Analysis     ', LinearDiscriminantAnalysis()))
models.append(('Random Forest Classifier         ', RandomForestClassifier()))
models.append(('KNeighbors Classifier            ', KNeighborsClassifier()))
models.append(('Decision Tree Classifier         ', DecisionTreeClassifier()))
models.append(('Gaussian Naive Bayes             ', GaussianNB()))
models.append(('Support vector machine Classifier', SVC()))
results = []
names = []
scoring = 'accuracy'
for (name, model) in models:
    kfold = model_selection.KFold(n_splits=10)
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())