import pandas as pd
dat = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_dat = pd.read_csv('data/input/spaceship-titanic/test.csv')
dat.head(3)
test_dat.head(3)
len(dat)
dat = dat.dropna()
test_dat = test_dat.dropna()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dat['HomePlanet'] = le.fit_transform(dat['HomePlanet'])
dat['CryoSleep'] = le.fit_transform(dat['CryoSleep'])
dat['Cabin'] = le.fit_transform(dat['Cabin'])
dat['Destination'] = le.fit_transform(dat['Destination'])
test_dat['HomePlanet'] = le.fit_transform(test_dat['HomePlanet'])
test_dat['CryoSleep'] = le.fit_transform(test_dat['CryoSleep'])
test_dat['Cabin'] = le.fit_transform(test_dat['Cabin'])
test_dat['Destination'] = le.fit_transform(test_dat['Destination'])
X_train = dat.iloc[:, 1:-2]
y_train = dat.iloc[:, -1]
X_test = test_dat.iloc[:, 1:-1]
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