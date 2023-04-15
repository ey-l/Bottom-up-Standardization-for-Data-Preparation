import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
trainDf = pd.read_csv('data/input/digit-recognizer/train.csv')
testDf = pd.read_csv('data/input/digit-recognizer/test.csv')
trainDf.head()
trainDf.shape
testDf.head()
testDf.shape
trainY = trainDf['label']
trainX = trainDf.drop(labels=['label'], axis=1)
plt.figure(figsize=(12, 5))
i = sns.countplot(trainY, palette='icefire')
plt.title('Numbers')
trainY.value_counts()