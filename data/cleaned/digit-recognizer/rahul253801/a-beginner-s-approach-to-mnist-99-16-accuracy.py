import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
mnist_df = pd.read_csv('data/input/digit-recognizer/train.csv')
mnist_df.head()
mnist_df.shape
import matplotlib as mpl
import matplotlib.pyplot as plt
for val in (10, 25, 1000, 2500):
    sample_digit = mnist_df.iloc[val][1:]
    sample_digit_mat = sample_digit.values.reshape(28, 28)
    plt.imshow(sample_digit_mat, cmap='binary')

for val in (10, 25, 1000, 2500):
    print(mnist_df.iloc[val][0])
mnist_df.isnull().any().describe()
import seaborn as sns
plt.ioff()
sns.set_theme(style='darkgrid')
ax = sns.countplot(data=mnist_df, x='label')
from sklearn.model_selection import train_test_split
X = mnist_df.drop(['label'], axis=1)
y = mnist_df['label']
print(X.shape)
print(y.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42, n_jobs=-1)
sgd_clf_log = SGDClassifier(random_state=42, n_jobs=-1, loss='log')