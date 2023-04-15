import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
sample = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
train.head()
train.label.value_counts().plot.bar()
im = np.reshape(train.iloc[np.random.randint(0, train.shape[0]), 1:].values, (28, 28))
from matplotlib import pyplot as plt
plt.imshow(im)
print(train.describe().T.sort_values(by='std', ascending=False))
train.isna().sum().values.sum()
train.iloc[:, 1:] = train.iloc[:, 1:] / 255
test = test / 255
print(train.describe().T.sort_values(by='std', ascending=False))
X = train.drop(['label'], axis=1)
y = train['label']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=42)
svm_model = svm.SVC(kernel='poly')