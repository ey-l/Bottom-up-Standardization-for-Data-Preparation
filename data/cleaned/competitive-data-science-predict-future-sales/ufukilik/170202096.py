import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.losses import MeanSquaredError
from keras.metrics import Recall, Precision, Accuracy
train = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
sample_submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
test = test.drop(['ID'], axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(train.iloc[:, 2:-2], train.iloc[:, -1], test_size=0.33, random_state=101)
X_train
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[Accuracy(), Precision(), Recall()])