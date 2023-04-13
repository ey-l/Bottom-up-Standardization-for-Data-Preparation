import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', header=None, names=col_names)
pima.head()
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]
y = pima.label
print('Total number of rows: {0}'.format(len(pima)))
print('Number of rows missing Pregnancies: {0}'.format(len(pima.loc[pima['pregnant'] == 0])))
print('Number of rows missing Glucose: {0}'.format(len(pima.loc[pima['glucose'] == 0])))
print('Number of rows missing BloodPressure: {0}'.format(len(pima.loc[pima['bp'] == 0])))
print('Number of rows missing SkinThickness: {0}'.format(len(pima.loc[pima['skin'] == 0])))
print('Number of rows missing Insulin: {0}'.format(len(pima.loc[pima['insulin'] == 0])))
print('Number of rows missing BMI: {0}'.format(len(pima.loc[pima['bmi'] == 0])))
print('Number of rows missing DiabetesPedigreeFunction: {0}'.format(len(pima.loc[pima['pedigree'] == 0])))
print('Number of rows missing Age: {0}'.format(len(pima.loc[pima['age'] == 0])))
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)