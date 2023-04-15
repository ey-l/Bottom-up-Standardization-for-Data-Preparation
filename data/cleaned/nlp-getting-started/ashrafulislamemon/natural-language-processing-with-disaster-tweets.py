import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
df2 = pd.read_csv('data/input/nlp-getting-started/test.csv')
df1
df1.drop_duplicates(inplace=True)
df1
x = df1['text']
y = df1['target']
x1 = df2['text']
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x_train = tf.fit_transform(x)
from sklearn.svm import LinearSVC
model = LinearSVC()