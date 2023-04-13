import pandas as pd
import matplotlib.pyplot as plt
_input0 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sales_train.csv')
_input2 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/test.csv')
_input5 = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
_input2 = _input2.drop(['ID'], axis=1)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(_input0.iloc[:, 2:-2], _input0.iloc[:, -1], test_size=0.33, random_state=101)
X_train