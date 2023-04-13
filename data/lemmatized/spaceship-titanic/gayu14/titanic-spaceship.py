import pandas as pd
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(3)
_input0.head(3)
len(_input1)
_input1 = _input1.dropna()
_input0 = _input0.dropna()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
_input1['HomePlanet'] = le.fit_transform(_input1['HomePlanet'])
_input1['CryoSleep'] = le.fit_transform(_input1['CryoSleep'])
_input1['Cabin'] = le.fit_transform(_input1['Cabin'])
_input1['Destination'] = le.fit_transform(_input1['Destination'])
_input0['HomePlanet'] = le.fit_transform(_input0['HomePlanet'])
_input0['CryoSleep'] = le.fit_transform(_input0['CryoSleep'])
_input0['Cabin'] = le.fit_transform(_input0['Cabin'])
_input0['Destination'] = le.fit_transform(_input0['Destination'])
X_train = _input1.iloc[:, 1:-2]
y_train = _input1.iloc[:, -1]
X_test = _input0.iloc[:, 1:-1]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=7)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)