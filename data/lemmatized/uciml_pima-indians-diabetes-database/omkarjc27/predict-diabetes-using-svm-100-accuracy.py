from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', sep=',')
dataset = df.values
positives = dataset[dataset[:, 8] == 1, :]
negatives = dataset[dataset[:, 8] == 0, :]
y = dataset[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(dataset[:, :], y, test_size=0.3, random_state=42)