import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pass
pass
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.isnull().sum()
pd.set_option('display.float_format', '{:.2f}'.format)
df.describe()
categorical_val = []
continous_val = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
df.columns
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for column in feature_columns:
    print('============================================')
    print(f'{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}')
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean', copy=False)
df[feature_columns] = fill_values.fit_transform(df[feature_columns])
for column in feature_columns:
    print('============================================')
    print(f'{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}')
from sklearn.model_selection import train_test_split
X = df[feature_columns]
y = df.Outcome
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print('TRAINIG RESULTS: \n===============================')
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f'CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}')
    print(f'ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}')
    print(f'CLASSIFICATION REPORT:\n{clf_report}')
    print('TESTING RESULTS: \n===============================')
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f'CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}')
    print(f'ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}')
    print(f'CLASSIFICATION REPORT:\n{clf_report}')
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=42)