import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.read_csv('data/input/digit-recognizer/train.csv')
test_data = pd.read_csv('data/input/digit-recognizer/test.csv')
LR_train = train_data.copy()
LR_test = test_data.copy()
CNN_train = train_data.copy()
CNN_test = test_data.copy()
LR_train.head(3)
LR_test.head(3)
print('Training: {} and Test : {}'.format(LR_train.shape, LR_test.shape))
train_y = LR_train['label']
LR_train.drop(['label'], axis=1, inplace=True)
LR_train.head(2)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(LR_train, train_y, test_size=0.3, random_state=42)
print('X train: {}'.format(x_train.shape))
print('Y train: {}'.format(y_train.shape))
print('X test: {}'.format(x_test.shape))
print('Y test: {}'.format(y_test.shape))
from sklearn.linear_model import LogisticRegression
regress = LogisticRegression(max_iter=500)
regress
import warnings
warnings.filterwarnings('ignore')