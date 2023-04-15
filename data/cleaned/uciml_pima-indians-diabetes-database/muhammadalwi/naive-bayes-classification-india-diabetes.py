import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data
data.head()
data.shape
data.describe()
X = data.drop('Outcome', axis=1)
y = data['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=17)
print(X_train.shape)
print(X_test.shape)
print(y_train.size)
print(y_test.size)
nb = LogisticRegression()