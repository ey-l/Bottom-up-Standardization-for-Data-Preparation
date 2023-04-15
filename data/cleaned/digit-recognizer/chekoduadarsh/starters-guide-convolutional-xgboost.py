import numpy as np
import pandas as pd
import seaborn as sns
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/digit-recognizer/train.csv')
print(data.shape)
data
test_data = pd.read_csv('data/input/digit-recognizer/test.csv')
print(test_data.shape)
test_data
sample_submission = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
print(sample_submission.shape)
sample_submission
train_data = data[:]
val_data = data[40000:]
train_label = np.float32(train_data.label)
val_label = np.float32(val_data.label)
train_image = np.float32(train_data[train_data.columns[1:]])
val_image = np.float32(val_data[val_data.columns[1:]])
test_image = np.float32(test_data[test_data.columns])
print('train shape: %s' % str(train_data.shape))
print('val shape: %s' % str(val_data.shape))
print('train_label shape: %s' % str(train_label.shape))
print('val_label shape: %s' % str(val_label.shape))
print('train_image shape: %s' % str(train_image.shape))
print('val_image shape: %s' % str(val_image.shape))
print('test_image shape: %s' % str(test_image.shape))
g = sns.countplot(train_label)
g
plt.imshow(train_image[13].reshape(28, 28))

print(train_image[13].shape)
train_image = train_image / 255.0
val_image = val_image / 255.0
test_image = test_image / 255.0
train_image = train_image.reshape(train_image.shape[0], 28, 28, 1)
val_image = val_image.reshape(val_image.shape[0], 28, 28, 1)
test_image = test_image.reshape(test_image.shape[0], 28, 28, 1)
print('train_image shape: %s' % str(train_image.shape))
print('train_image shape: %s' % str(train_image.shape))
print('val_image shape: %s' % str(val_image.shape))
train_label1 = train_label
val_label1 = val_label
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, categories='auto')
yy = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]