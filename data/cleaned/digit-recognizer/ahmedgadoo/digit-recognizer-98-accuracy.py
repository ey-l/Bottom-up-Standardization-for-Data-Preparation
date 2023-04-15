import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
plt.figure(figsize=(15, 15))
sns.set_style('darkgrid')
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize, Binarizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
digits_data = pd.read_csv('data/input/digit-recognizer/train.csv')
digits_test = pd.read_csv('data/input/digit-recognizer/test.csv')
digits_submit = pd.read_csv('data/input/digit-recognizer/sample_submission.csv')
digits_data
digits_data.columns[digits_data.isna().sum() > 0]
print(digits_data['label'].value_counts().sort_index())
(fig, axes) = plt.subplots(nrows=10, ncols=6, figsize=(20, 40), sharey=True, sharex=True)
idx = 0
for i in range(10):
    for j in digits_data[digits_data.label == i].sample(n=6).index:
        digit = digits_data['label'].loc[j]
        idx += 1
        plt.subplot(10, 6, idx)
        plt.imshow(np.array(digits_data.iloc[j, 1:]).reshape(28, 28), cmap='cividis')
        plt.title(f'Digit is {digit}', fontsize=14)
        plt.grid(None)

for k in range(10):
    temp_freq = pd.Series(digits_data.iloc[:, 1:].sample(1).T.value_counts(), name='freq')
    print(temp_freq[temp_freq > 3])
    print('=' * 30)
sns.displot(data=digits_data.iloc[:, 1:].sample(100).T, kind='kde', legend=False)

(fig, axes) = plt.subplots(nrows=10, ncols=6, figsize=(20, 40), sharey=True, sharex=True)
idx = 0
for k in np.random.choice(range(42000), 10):
    img = np.array(digits_data.iloc[k, 1:]).reshape(28, 28)
    idx += 1
    plt.grid(None)
    plt.subplot(10, 6, idx)
    plt.imshow(img, cmap='Accent')
    plt.title(f'Original Image\n Digit={digits_data.iloc[k, 0]}', fontsize=14)
    for i in range(1, 6):
        idx += 1
        plt.grid(None)
        plt.subplot(10, 6, idx)
        plt.imshow(binarize(img, threshold=i * 50), cmap='Accent')
        plt.title(f'Binarize at\n threshold={i * 50}', fontsize=14)

digitsX = digits_data.iloc[:, 1:]
digitsY = digits_data.label