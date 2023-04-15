import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
train_file = 'data/input/digit-recognizer/train.csv'
test_file = 'data/input/digit-recognizer/test.csv'
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
df_train.describe()
df_test.describe()
(df_train.shape, df_test.shape)
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values
X_test = df_test.values
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)
(X_train_partial, X_holdout, y_train_partial, y_holdout) = train_test_split(X_train, y_train, train_size=0.75, random_state=99)
pca = PCA(n_components=X_train.shape[1])