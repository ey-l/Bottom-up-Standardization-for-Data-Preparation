import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/digit-recognizer/train.csv')
df_test = pd.read_csv('data/input/digit-recognizer/test.csv')
df_train
df_train.isnull().sum() / len(df_train.index)
df_train.describe()
train_images = df_train.iloc[:, 1:].values.reshape(-1, 28, 28)
plt.imshow(train_images[25], cmap='gray')
plt.imshow(train_images[100], cmap='gray')
df_train['label'].value_counts()
list(df_train['label'].unique())

def generate_sample(df, label, fraction=0.2):
    classes = list(df[label].unique())
    df_sample = pd.DataFrame(columns=df.columns)
    for cl in classes:
        df1 = df[df[label] == cl].sample(frac=fraction)
        df_sample = df_sample.append(df1)
    return df_sample
df_sample_train = generate_sample(df_train, 'label', 0.1)
X = df_sample_train.iloc[:, 1:]
y = df_sample_train['label']
y = y.astype('int')
X = X / 255.0
df_test = df_test / 255
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(fit_intercept=True, multi_class='auto', penalty='l2', solver='saga', C=50)