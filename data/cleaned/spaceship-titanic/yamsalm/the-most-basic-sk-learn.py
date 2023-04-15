import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_data = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_data = pd.read_csv('data/input/spaceship-titanic/test.csv')
train_data.head()
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
train_data.columns
num_columns = train_data.select_dtypes(include=np.number).columns.tolist()
num_columns
X_train_val = train_data[num_columns]
X_test = test_data[num_columns]
y_train_val = train_data['Transported'].astype(int)
(X_train, X_val, y_train, y_val) = train_test_split(X_train_val, y_train_val, test_size=0.33, random_state=42)
PassengerID = test_data.PassengerId
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()