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
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
df_test = pd.read_csv('data/input/nlp-getting-started/test.csv')
print(df_train.head(5))
print(df_train.info())
print(df_train.isnull().any())
print(df_test.isnull().any())
print(df_train.shape)
import seaborn as sns
from matplotlib import pyplot as plt
(fig, axes) = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
plt.tight_layout()
labels = ['Disaster Tweet', 'No Disaster']
size = [df_train['target'].mean() * 100, abs(1 - df_train['target'].mean()) * 100]
explode = (0, 0.1)
axes[0].pie(size, labels=labels, explode=explode, shadow=True, startangle=90, autopct='%1.1f%%')
sns.countplot(x=df_train['target'], hue=df_train['target'], ax=axes[1])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
(X_train, X_test, y_train, y_test) = train_test_split(df_train['text'], df_train['target'])