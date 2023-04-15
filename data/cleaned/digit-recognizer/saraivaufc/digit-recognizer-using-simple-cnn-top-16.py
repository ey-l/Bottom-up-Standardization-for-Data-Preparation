import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df
y_data = df.pop('label')
y_data
import numpy as np
from sklearn import preprocessing

def transform_labels(labels_array, labels):
    lb = preprocessing.LabelBinarizer()