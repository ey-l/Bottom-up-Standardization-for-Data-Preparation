import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
train_data = pd.read_csv('data/input/digit-recognizer/train.csv')
test_data = pd.read_csv('data/input/digit-recognizer/test.csv')
sample = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
X_train = train_data.iloc[:, 1:]
y_train = train_data['label']