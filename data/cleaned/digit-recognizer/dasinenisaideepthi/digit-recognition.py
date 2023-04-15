import numpy as np
import pandas as pd
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
df_train = pd.read_csv('data/input/digit-recognizer/train.csv')
df_test = pd.read_csv('data/input/digit-recognizer/test.csv')
df_train.head()
X_train = df_train.drop('label', axis=1)
y_train = df_train['label']
X_test = df_test
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
X_train = X_train.reshape((-1, 28, 28))
X_test = X_test.reshape((-1, 28, 28))
import seaborn as sns
sns.countplot(x=y_train)
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_train[i])

X_train = X_train.reshape((-1, 28 * 28))
X_test = X_test.reshape((-1, 28 * 28))
model = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64), early_stopping=True, verbose=True)