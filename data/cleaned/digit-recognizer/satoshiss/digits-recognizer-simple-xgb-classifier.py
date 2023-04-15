import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train = pd.read_csv('data/input/digit-recognizer/train.csv')
df_test = pd.read_csv('data/input/digit-recognizer/test.csv')
df_train
df_train.label.unique()
from matplotlib.pyplot import imshow
width = 5
height = 5
rows = 2
cols = 3
axes = []
fig = plt.figure()
fig.set_size_inches(8, 10)
for i in range(rows * cols):
    sample = np.reshape(df_train[df_train.columns[1:]].iloc[i].values / 255, (28, 28))
    axes.append(fig.add_subplot(rows, cols, i + 1))
    plt.title('Labeled class : {}'.format(df_train['label'].iloc[i]))
    plt.imshow(sample, 'gray')
fig.tight_layout()

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='label', data=df_train)
plt.title('Label Distribution')
total = len(df_train.label)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
df_train.describe()
df_train.sum(axis=1)
df_train.shape
pixels = df_train.columns.tolist()[1:]
df_train['sum'] = df_train[pixels].sum(axis=1)
df_test['sum'] = df_test[pixels].sum(axis=1)
df_train.groupby(['label'])['sum'].mean()
len(df_train)
targets = df_train.label
features = df_train.drop('label', axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features[:] = scaler.fit_transform(features)
df_test[:] = scaler.transform(df_test)
del df_train
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(features)
Y_sklearn
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 8))
    for (lab, col) in zip((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), ('blue', 'red', 'green', 'yellow', 'purple', 'black', 'brown', 'pink', 'orange', 'beige')):
        plt.scatter(Y_sklearn[targets == lab, 0], Y_sklearn[targets == lab, 1], label=lab, c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower right')
    plt.tight_layout()

features.index
sklearn_pca_3 = sklearnPCA(n_components=3)
Y_sklearn_3 = sklearn_pca_3.fit_transform(features)
Y_sklearn_3_test = sklearn_pca_3.transform(df_test)
result = pd.DataFrame(Y_sklearn_3, columns=['PCA%i' % i for i in range(3)], index=features.index)
result
my_dpi = 96
plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)
with plt.style.context('seaborn-whitegrid'):
    my_dpi = 96
    fig = plt.figure(figsize=(10, 10), dpi=my_dpi)
    ax = fig.add_subplot(111, projection='3d')
    for (lab, col) in zip((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), ('blue', 'red', 'green', 'yellow', 'purple', 'black', 'brown', 'pink', 'orange', 'beige')):
        plt.scatter(Y_sklearn[targets == lab, 0], Y_sklearn[targets == lab, 1], label=lab, c=col, s=60)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('PCA on the Handwriting Data')

encoder = LabelEncoder()
targets[:] = encoder.fit_transform(targets[:])
(X_train, X_val, y_train, y_val) = train_test_split(result, targets, random_state=1)
model = XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000, num_classes=10)