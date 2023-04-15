import numpy as np
import pandas as pd
import os
from sklearn import model_selection as sk_model_selection
from sklearn.feature_extraction import text as sk_fe_text
from sklearn import svm as sk_svm
from sklearn import metrics as sk_metrics
base_dir = 'data/input/nlp-getting-started/'
df_train = pd.read_csv(os.path.join(base_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
df_submission = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
print(f'df_train shape: {df_train.shape}')
df_train.head()
df_train.isna().sum()
X_train = df_train['text']
y_train = df_train['target'].values
tfidf = sk_fe_text.TfidfVectorizer(stop_words='english')