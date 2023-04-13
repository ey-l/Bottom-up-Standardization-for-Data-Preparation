import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter('ignore')
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.width', 1000000)
pd.set_option('display.max_columns', 500)
score_df = pd.DataFrame(columns={'Model Description', 'Score'})
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
print(_input1.head(5))
print(_input1.info())
print(_input1.isnull().any())
print(_input0.isnull().any())
print(_input1.shape)
import seaborn as sns
from matplotlib import pyplot as plt
(fig, axes) = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
plt.tight_layout()
labels = ['Disaster Tweet', 'No Disaster']
size = [_input1['target'].mean() * 100, abs(1 - _input1['target'].mean()) * 100]
explode = (0, 0.1)
axes[0].pie(size, labels=labels, explode=explode, shadow=True, startangle=90, autopct='%1.1f%%')
sns.countplot(x=_input1['target'], hue=_input1['target'], ax=axes[1])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
(X_train, X_test, y_train, y_test) = train_test_split(_input1['text'], _input1['target'])