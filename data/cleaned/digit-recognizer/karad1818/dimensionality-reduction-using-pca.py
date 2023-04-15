import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from sklearn import decomposition
from sklearn import preprocessing
data = pd.read_csv('data/input/digit-recognizer/train.csv')
data.head()
label = data.label
data = data.drop('label', axis=1)
one_image = data.loc[3, :]
plt.figure(figsize=(7, 7))
one = np.array(one_image).reshape(28, 28)
plt.imshow(one, cmap='gray')
scaler = preprocessing.StandardScaler()