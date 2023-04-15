import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
idnum = pd.read_csv('data/input/digit-recognizer/train.csv')
idnum
idnum.info()
idnum.isnull().sum()
df_label = idnum.groupby('label').count()['pixel0']
plt.figure(figsize=(7, 7))
plt.title('Label distribution')
plt.pie(df_label, labels=df_label.index, autopct='%0.0f%%')
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(idnum.iloc[i, 1:].values.reshape(28, 28), cmap='gray')
idtest = pd.read_csv('data/input/digit-recognizer/test.csv')
idtest
idtest.info()
x = idnum.drop(columns=['label'])
y = idnum['label']
model = mlp()