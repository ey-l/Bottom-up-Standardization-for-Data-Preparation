import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
X_train = train.iloc[:, 1:]
y_train = train.label