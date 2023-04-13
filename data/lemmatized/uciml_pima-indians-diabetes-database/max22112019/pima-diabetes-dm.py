import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
import seaborn as sns
pass
diabetes_count = len(df.loc[df['Outcome'] == 1])
no_diabetes_count = len(df.loc[df['Outcome'] == 0])
(diabetes_count, no_diabetes_count)
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
num = df[cols]
for i in num.columns:
    pass
    pass
print('total number of rows : {0}'.format(len(df)))
print('number of rows with 0 Pregnancies: {0}'.format(len(df.loc[df['Pregnancies'] == 0])))
print('number of rows with 0 Glucose: {0}'.format(len(df.loc[df['Glucose'] == 0])))
print('number of rows with 0 BloodPressure: {0}'.format(len(df.loc[df['BloodPressure'] == 0])))
print('number of rows with 0 SkinThickness: {0}'.format(len(df.loc[df['SkinThickness'] == 0])))
print('number of rows with 0 Insulin: {0}'.format(len(df.loc[df['Insulin'] == 0])))
print('number of rows with 0 BMI: {0}'.format(len(df.loc[df['BMI'] == 0])))
print('number of rows with 0 DiabetesPedigreeFunction: {0}'.format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print('number of rows with 0 Ages: {0}'.format(len(df.loc[df['Age'] == 0])))
from sklearn.impute import SimpleImputer
zcol = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zcols = df[zcol]
imputer = SimpleImputer(missing_values=0, strategy='mean', verbose=0)
imputed_df = pd.DataFrame(imputer.fit_transform(zcols))
imputed_df.columns = zcols.columns
temp = imputed_df.copy()
zcols = temp.copy()
zcols.head()
df.drop(['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'], axis=1, inplace=True)
df = df.join(zcols)
df.head()
df.dtypes
outcome = df['Outcome']
df.drop(['Outcome'], axis=1, inplace=True)
df = df.join(outcome)
df.head()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X.head()
y.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
X_train.head()
X_test.head()
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=8)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression