import pandas as pd
import numpy as np
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
x = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(x, y, test_size=0.2, random_state=31)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression