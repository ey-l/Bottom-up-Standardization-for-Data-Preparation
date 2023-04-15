import numpy as np
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
ss = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
print('Number of Missing Values in Target feature: {}'.format(train.target.isnull().sum()))
(canv, axs) = plt.subplots(1, 2, figsize=(22, 8))
color = ['darkgreen', 'darkslategrey']
plt.sca(axs[0])
plt.pie(train.groupby('target').count()['id'], explode=(0.1, 0), startangle=120, colors=color, textprops={'fontsize': 15}, labels=['Not Disaster (57%)', 'Disaster (43%)'])
plt.sca(axs[1])
bars = plt.bar([0, 0.5], train.groupby('target').count()['id'], width=0.3, color=color)
plt.xticks([0, 0.5], ['Not Disaster', 'Disaster'])
plt.tick_params(axis='both', labelsize=15, size=0, labelleft=False)
for sp in plt.gca().spines.values():
    sp.set_visible(False)
for (bar, val) in zip(bars, train.groupby('target').count()['id']):
    plt.text(bar.get_x() + 0.113, bar.get_height() - 250, val, color='w', fontdict={'fontsize': 18, 'fontweight': 'bold'})
canv.suptitle('Target Value Distribution in Training Data', fontsize=18)
train_na = train.isnull().sum() / len(train) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
pd.DataFrame({'Train Missing Ratio': train_na}).head(3)
test_na = test.isnull().sum() / len(test) * 100
test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)
pd.DataFrame({'Test Missing Ratio': test_na}).head(3)
title = 'Train'
data = [train_na, test_na]
(canv, axs) = plt.subplots(1, 2)
canv.set_size_inches(18, 5)
for (ax, dat) in zip(axs, data):
    plt.sca(ax)
    sns.barplot(x=dat.index, y=dat, dodge=False)
    plt.xlabel('Features', fontsize=15, labelpad=10)
    plt.ylabel('Percent of missing values', fontsize=15, labelpad=13)
    plt.title('Percent missing data by feature in {} Data'.format(title), fontsize=15, pad=20)
    plt.tick_params(axis='both', labelsize=12)
    sp = plt.gca().spines
    sp['top'].set_visible(False)
    sp['right'].set_visible(False)
    title = 'Test'
for df in [train, test]:
    for col in ['keyword', 'location']:
        df[col].fillna('None', inplace=True)

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
y = train.target
X = train.text
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.2)
from sklearn.feature_extraction.text import TfidfVectorizer