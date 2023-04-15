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
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.head()
train_data.describe()
train_data.info()
train_data.corr()
sns.heatmap(train_data.corr(), cmap='GnBu')
data = pd.DataFrame(train_data.isnull().sum(), columns=['Number of null'])
data['percentage'] = train_data.isnull().sum() / len(train_data) * 100
data
train_data.drop('Name', axis=1, inplace=True)
train_data.drop('PassengerId', axis=1, inplace=True)
index = test_data['PassengerId']
test_data.drop('Name', axis=1, inplace=True)
test_data.drop('PassengerId', axis=1, inplace=True)
train_data.drop('Destination', axis=1, inplace=True)
test_data.drop('Destination', axis=1, inplace=True)
train_data.duplicated().sum()
sns.countplot(x='Transported', data=train_data)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
X_key = X.keys()
X
y
label = LabelEncoder()
y = label.fit_transform(y)
pd.DataFrame(y)
list = ['HomePlanet', 'CryoSleep', 'Cabin', 'VIP']
for col in list:
    X[col] = label.fit_transform(X[col])
    test_data[col] = label.fit_transform(test_data[col])
X
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
X = impute.fit_transform(X)
test_data = impute.fit_transform(test_data)
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