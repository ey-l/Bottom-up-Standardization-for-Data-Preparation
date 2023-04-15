import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from collections import Counter
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
training_data = pd.read_csv('data/input/digit-recognizer/train.csv')
train_label = train['label']
train
train.shape

def create_pie(df, target_variable, figsize=(10, 10)):
    print(df[target_variable].value_counts())
    (fig, ax) = plt.subplots(figsize=figsize)
    ax.pie(df[target_variable].value_counts().values, labels=df[target_variable].value_counts().index, autopct='%1.2f%%', textprops={'fontsize': 10})
    ax.axis('equal')
    plt.title(target_variable)

plt.figure(figsize=(25, 25))
create_pie(train, 'label')
cp = train.drop(['label'], axis=1)
ratio = int(math.sqrt(cp.shape[1]))
train = train.drop(['label'], axis=1)
image_first = train.iloc[10]
image_first = np.array(image_first).reshape(ratio, ratio)
plt.imshow(image_first)
plt.figure(figsize=(25, 25))
columns = 3
firsts_image = train.iloc[:10]
for i in range(0, 9):
    image = np.array(train.iloc[i]).reshape(ratio, ratio)
    plt.subplot(int(firsts_image.shape[0] / columns + 1), columns, i + 1)
    plt.imshow(image, cmap='Greens')
X = train
y = train_label.to_list()
X = train[:1000]
y = train_label[:1000].to_list()
X = X / 255.0
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=42)
(X_test, X_val, y_test, y_val) = train_test_split(X_test, y_test, test_size=0.3, random_state=42)
param_grid = {'kernel': ['linear', 'poly'], 'degree': [1, 2], 'gamma': [0.01, 0.1], 'coef0': [0.5, 1]}
grid = GridSearchCV(svm.SVC(), param_grid, cv=3)