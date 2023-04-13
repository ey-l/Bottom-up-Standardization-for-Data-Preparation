import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
train_file = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
X_train = pd.read_csv(train_file)
print('Shape of the training data: ', X_train.shape)
print('- ' * 70)
print(X_train.info())
print('- ' * 70)
X_train.describe(include='all')
missing_val_count_by_column_train = X_train.isnull().sum()
print(missing_val_count_by_column_train[missing_val_count_by_column_train > 0])
pass
diabetes_data_copy = X_train.copy()
diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(diabetes_data_copy.isnull().sum())
pass
p = X_train.hist(figsize=(20, 20))
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace=True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace=True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace=True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace=True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace=True)
p = diabetes_data_copy.hist(figsize=(20, 20))
pass
pass
sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(['Outcome'], axis=1)), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = diabetes_data_copy.Outcome
X.head()
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
test_scores = []
train_scores = []
for i in range(1, 27):
    knn = KNeighborsClassifier(i)