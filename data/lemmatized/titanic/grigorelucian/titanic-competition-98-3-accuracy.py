import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
_input1 = pd.read_csv('data/input/titanic/train.csv')
_input0 = pd.read_csv('data/input/titanic/test.csv')
_input2 = pd.read_csv('data/input/titanic/gender_submission.csv')
x_train = _input1.iloc[:, [2, 4, 5, 6, 7]].values
y_train = _input1.iloc[:, 1].values
x_test = _input0.iloc[:, [1, 3, 4, 5, 6]].values
y_test = _input2.iloc[:, 1].values
le1 = LabelEncoder()
x_train[:, 1] = le1.fit_transform(x_train[:, 1])
x_test[:, 1] = le1.fit_transform(x_test[:, 1])
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train_imp = imp.fit_transform(x_train)
x_test_imp = imp.fit_transform(x_test)
sc = StandardScaler()
x_train_imp = sc.fit_transform(x_train_imp)
x_test_imp = sc.transform(x_test_imp)
rfc = RandomForestClassifier(n_estimators=10000, criterion='entropy', n_jobs=-1)