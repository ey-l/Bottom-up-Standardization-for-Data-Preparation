import pandas as pd
from sklearn.model_selection import train_test_split
dg = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dg.head()
glucose = dg.iloc[:, 1].values
bp = dg.iloc[:, 2].values
X = dg.iloc[:, ::-1].values
y = dg.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X = le.fit_transform(glucose)
X = X.reshape(-1, 1)
y = le.fit_transform(bp)
y = y.reshape(-1, 1)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()