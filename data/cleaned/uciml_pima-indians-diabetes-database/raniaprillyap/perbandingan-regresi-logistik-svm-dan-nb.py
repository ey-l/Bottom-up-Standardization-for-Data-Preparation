import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
data1 = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')[:500]
X1 = data1.drop(['Outcome'], axis=1)
y1 = data1['Outcome']
from sklearn.model_selection import train_test_split
(X_train1, X_test1, y_train1, y_test1) = train_test_split(X1, y1, test_size=0.3, random_state=0)
(X_train1.shape, X_test1.shape)
cols1 = X_train1.columns
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.fit_transform(X_test1)
X_train1 = pd.DataFrame(X_train1, columns=[cols1])
X_train1.head()
X_test1 = pd.DataFrame(X_test1, columns=[cols1])
X_test1.head()
logreg = LogisticRegression()
svm = SVC(probability=True)
nb = GaussianNB()