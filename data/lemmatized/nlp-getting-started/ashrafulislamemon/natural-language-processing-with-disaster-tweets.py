import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1
_input1 = _input1.drop_duplicates(inplace=False)
_input1
x = _input1['text']
y = _input1['target']
x1 = _input0['text']
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x_train = tf.fit_transform(x)
from sklearn.svm import LinearSVC
model = LinearSVC()