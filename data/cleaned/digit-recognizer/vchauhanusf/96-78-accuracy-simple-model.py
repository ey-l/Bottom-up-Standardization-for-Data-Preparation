import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df
y = df.iloc[:, 0]
y
y.value_counts()
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
x = df.iloc[:, 1:]
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=1e-05, random_state=2)