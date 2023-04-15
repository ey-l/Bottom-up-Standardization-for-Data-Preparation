import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import shift, rotate, zoom

for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('data/input/digit-recognizer/train.csv')
test_df = pd.read_csv('data/input/digit-recognizer/test.csv')
submission_df = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
train_df.info()
test_df.info()
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.values
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
some_digit = X_train[40]
some_digit_image = some_digit.reshape(28, 28)
print(f'Label: {y_train[40]}')
plt.imshow(some_digit_image, cmap='binary')

stratified_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for (fold, indices) in enumerate(stratified_fold.split(X_train, y_train)):
    (X_train_, y_train_) = (X_train[indices[0]], y_train[indices[0]])
    (X_test_, y_test_) = (X_train[indices[1]], y_train[indices[1]])
    estimator = KNeighborsClassifier()