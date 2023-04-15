import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()

df['Outcome'].value_counts()
split = int(768 * 0.7)
(train, test) = (df.iloc[:split, :], df.iloc[split:, :])
(train.shape, test.shape)
(trainX, trainY) = (train.iloc[:, :-1].to_numpy(), train.iloc[:, -1].to_numpy())
(trainX.shape, trainY.shape, type(trainX))
trainY = trainY.reshape((trainY.shape[0], 1))
trainY.shape
trainX = np.c_[trainX, np.ones((trainX.shape[0], 1))]
trainX.shape
np.linalg.det(trainX.T.dot(trainX))
weights = np.linalg.inv(trainX.T.dot(trainX)).dot(trainX.T).dot(trainY)
weights.shape
(testX, testY) = (test.iloc[:, :-1].to_numpy(), test.iloc[:, -1].to_numpy())
(testX.shape, testY.shape, type(testX))
testY = testY.reshape((testY.shape[0], 1))
testX = np.c_[testX, np.ones((testX.shape[0], 1))]
(testX.shape, testY.shape)

def threshold(i):
    if i < 0.5:
        return 0
    else:
        return 1
preds = testX.dot(weights)
result = np.vectorize(threshold)(preds)
sum(testY == result)
sum(testY == result) / testY.shape[0] * 100