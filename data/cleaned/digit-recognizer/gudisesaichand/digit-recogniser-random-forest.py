import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df.head(n=5)
df.info()
df.describe()
df.isnull().sum()
df.drop_duplicates()
x = df.drop(['label'], axis=1)
y = df['label']
five = df.iloc[df[df['label'] == 5].index[0], 1:].values.reshape(28, 28)
plt.imshow(five, cmap='gray')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
(x_train, x_val, y_train, y_val) = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
from sklearn.ensemble import RandomForestClassifier
r_clf = RandomForestClassifier()