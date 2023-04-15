import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('data/input/digit-recognizer/train.csv')
test = pd.read_csv('data/input/digit-recognizer/test.csv')
sample_submission = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
label = train.iloc[:, 0].values
print('Training Dataset:\n')
print('The Shape is: ', train.shape)
print('\nSome Samples:\n', train.sample(5))
print('\nThere is ', str((train.isna().sum() > 0).sum()), ' null values in the training dataset')
print('-' * 80)
print('Testing Dataset:\n')
print('The Shape is: ', test.shape)
print('\nSome Samples:\n', test.sample(5))
print('\nThere is ', str((test.isna().sum() > 0).sum()), ' null values in the testing dataset')
plt.hist(train['label'], bins=10)
plt.title('The labels Frequencies')
plt.xlabel('Label')
plt.ylabel('Frequency')

i = np.random.randint(0, train.shape[0], 1)
plt.hist(train.iloc[i, 1:])
plt.title('Pixel range')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

train_without_label = train.drop('label', axis=1)
from sklearn.decomposition import PCA
pca = PCA()