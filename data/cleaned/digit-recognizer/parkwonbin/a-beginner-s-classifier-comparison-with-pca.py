import pandas as pd
train_set = pd.read_csv('data/input/digit-recognizer/train.csv')
X = train_set.drop('label', axis=1)
y = train_set['label']
import numpy as np
num = int(len(X) * (3 / 5))
(X_train, X_valid) = (X[:num], X[num:])
(y_train, y_valid) = (y[:num], y[num:])
print('X_train :', len(X_train))
print('y_train :', len(y_train))
print('X_valid :', len(X_valid))
print('y_valid :', len(y_valid))
print('X_train :', len(X_train))
print('y_train :', len(y_train))
print('X_valid :', len(X_valid))
print('y_valid :', len(y_valid))
import time
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

def train(*models, dataset=(X_train, y_train, X_valid, y_valid)):
    columns = ['Name', 'Time(sec)', 'accuracy(%)', 'precision(%)', 'recall(%)', 'f1-score', 'confusion', 'model']
    df = pd.DataFrame(columns=columns)
    (X_train, y_train, X_valid, y_valid) = dataset
    for model in tqdm(models):
        model_name = str(model.__class__.__name__)
        print(model_name, end='...')
        start = time.time()