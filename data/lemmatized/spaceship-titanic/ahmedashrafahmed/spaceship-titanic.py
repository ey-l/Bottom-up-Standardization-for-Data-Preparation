import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head()
_input1.describe()
_input1.info()
_input1.corr()
sns.heatmap(_input1.corr(), cmap='GnBu')
data = pd.DataFrame(_input1.isnull().sum(), columns=['Number of null'])
data['percentage'] = _input1.isnull().sum() / len(_input1) * 100
data
_input1 = _input1.drop('Name', axis=1, inplace=False)
_input1 = _input1.drop('PassengerId', axis=1, inplace=False)
index = _input0['PassengerId']
_input0 = _input0.drop('Name', axis=1, inplace=False)
_input0 = _input0.drop('PassengerId', axis=1, inplace=False)
_input1 = _input1.drop('Destination', axis=1, inplace=False)
_input0 = _input0.drop('Destination', axis=1, inplace=False)
_input1.duplicated().sum()
sns.countplot(x='Transported', data=_input1)
X = _input1.iloc[:, :-1]
y = _input1.iloc[:, -1]
X_key = X.keys()
X
y
label = LabelEncoder()
y = label.fit_transform(y)
pd.DataFrame(y)
list = ['HomePlanet', 'CryoSleep', 'Cabin', 'VIP']
for col in list:
    X[col] = label.fit_transform(X[col])
    _input0[col] = label.fit_transform(_input0[col])
X
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
X = impute.fit_transform(X)
_input0 = impute.fit_transform(_input0)
X = pd.DataFrame(X, columns=X_key)
X
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)
print('X \n', X[:10])
print('y \n', y[:10])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=33)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
RandomForestClassifierModel = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=15, random_state=33)