import numpy as np
import pandas as pd
import matplotlib as ml
import matplotlib.pyplot as plt
import seaborn as sns

ml.style.use('fivethirtyeight')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head(10)
data.info()
data.describe()
X = np.array(data.iloc[:, :-1].values)
Y = np.array(data.iloc[:, -1].values)
(trainx, testx, trainy, testy) = train_test_split(X, Y, test_size=0.2, random_state=0)
sc = StandardScaler()
(trainx, testx) = (sc.fit_transform(trainx), sc.fit_transform(testx))
print('For X : Training : {} ; Testing : {}'.format(trainx.shape, testx.shape))
print('\nFor Y : Training : {} ; Testing : {}'.format(trainy.shape, testy.shape))
sns.pairplot(data, hue='Outcome')

m = 50
(accuracy, recall, f1) = ([], [], [])
x_axis = [i for i in range(1, m)]
for i in range(1, m):
    svm_c = SVC(kernel='rbf', C=i, gamma='auto')