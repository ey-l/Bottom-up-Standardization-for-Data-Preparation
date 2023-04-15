import pandas as pd
import numpy as np
train = pd.read_csv('data/input/spaceship-titanic/train.csv')
test = pd.read_csv('data/input/spaceship-titanic/test.csv')
train.head(3)
train = train.drop(['PassengerId', 'Name'], axis=1)
test = test.drop(['PassengerId', 'Name'], axis=1)
test.shape
train.shape
train.info()
test.info()
train[['deck', 'num', 'side']] = train['Cabin'].str.split('/', expand=True)
test[['deck', 'num', 'side']] = test['Cabin'].str.split('/', expand=True)
train = train.drop(['Cabin', 'num'], axis=1)
test = test.drop(['Cabin', 'num'], axis=1)
pd.crosstab(train.Transported, train.deck)
pd.crosstab(train.Transported, train.side)
pd.crosstab(train.Transported, train.HomePlanet)
pd.crosstab(train.Transported, train.Destination)
pd.crosstab(train.Transported, train.VIP)
pd.crosstab(train.Transported, train.Destination)
pd.crosstab(train.Transported, train.CryoSleep)
train_df = pd.get_dummies(train, drop_first=True)
test_df = pd.get_dummies(test, drop_first=True)
test_df.shape
from sklearn import preprocessing
label_encoder1 = preprocessing.LabelEncoder()
label_encoder2 = preprocessing.LabelEncoder()
train_df['Transported'] = label_encoder1.fit_transform(train_df['Transported'])
train_df.head(5)
train_df.isna().sum()
test_df.isna().sum()
df = train_df.copy()
from scipy.stats import chi2_contingency
import numpy as np
chisqt = pd.crosstab(df.VIP_True, df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values, chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print('x2 , p value , degree of freedom.')
from scipy.stats import chi2_contingency
import numpy as np
chisqt = pd.crosstab(df.CryoSleep_True, df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values, chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print('x2 , p value , degree of freedom.')
from scipy.stats import chi2_contingency
import numpy as np
chisqt = pd.crosstab(df.HomePlanet_Europa, df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values, chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print('x2 , p value , degree of freedom.')
from scipy.stats import chi2_contingency
import numpy as np
chisqt = pd.crosstab(df.HomePlanet_Mars, df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values, chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print('x2 , p value , degree of freedom.')
train_df.info()
test_df.info()
train_df.shape
test_df.shape
a = train_df.select_dtypes(['float64']).fillna(train_df.select_dtypes(['float64']).mean())
b = train_df.select_dtypes(['uint8']).fillna(train_df.select_dtypes(['uint8']).mode())
kümelemetrain = pd.concat([a, b], axis=1)
c = test_df.select_dtypes(['float64']).fillna(test_df.select_dtypes(['float64']).mean())
d = test_df.select_dtypes(['uint8']).fillna(test_df.select_dtypes(['uint8']).mode())
kümelemetest = pd.concat([c, d], axis=1)
test_df.head()
kümelemetrain.shape
kümelemetrain.isna().sum()
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mntrain = StandardScaler()
mntest = StandardScaler()
x1 = kümelemetrain.select_dtypes('float64')
x2 = kümelemetest.select_dtypes('float64')
x1.head(2)
x2.head(2)
x1 = mntrain.fit_transform(x1)
x2 = mntest.fit_transform(x2)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, init='k-means++')