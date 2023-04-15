#!/usr/bin/env python
# coding: utf-8

# This Notebook is a kaggle tutorial for Japanese kaggle beginners writen in Japanese.
# 
# # 3. ここで差がつく！ 仮説に基づいて新しい特徴量を作ってみよう

# この[Notebook](https://www.kaggle.com/sishihara/upura-kaggle-tutorial-03-feature-engineering)では、特徴量エンジニアリングを学びます。

# # 再現性の大切さ
# 「再現性がある」とは、何度実行しても同じ結果が得られることです。Kaggleで言うと、同一のスコアが得られると言い換えても良いでしょう。
# 
# 再現性がないと、実行ごとに異なるスコアが得られてしまいます。今後、特徴量エンジニアリングなどでスコアの向上を試みても、予測モデルが改善されたか否かを正しく判断できなくなる問題が生じます。
# 
# 実は、2つ目のNotebookには再現性がありませんでした。その原因は、Ageという特徴量の欠損値を埋める際の乱数です。ここでは標準偏差を考慮した乱数で欠損値を穴埋めしているのですが、この乱数は実行ごとに値が変わるようになってしまっています。

# In[ ]:


# 前回のAgeを処理する部分までを実行

import numpy as np
import pandas as pd

train = pd.read_csv("data/input/titanic/train.csv")
test = pd.read_csv("data/input/titanic/test.csv")
gender_submission = pd.read_csv("data/input/titanic/gender_submission.csv")

data = pd.concat([train, test], sort=False)

data['Sex'].replace(['male','female'], [0, 1], inplace=True)
data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)


# `np.random.randint(age_avg - age_std, age_avg + age_std)` の実行ごとに、結果が異なります。

# In[ ]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()

np.random.randint(age_avg - age_std, age_avg + age_std)


# In[ ]:


np.random.randint(age_avg - age_std, age_avg + age_std)


# 再現性を確保するためには、例えば次のような方法が考えられます。
# 
# 1. そもそも乱数を用いる部分を削除する
# 2. 乱数のseedを与えて実行結果を固定する
# 
# Ageについては、そもそも乱数を用いるよりも、欠損していないデータの中央値を与えた方が筋の良い補完ができそうです。今回は中央値で補完するようにコードを改変します。

# In[ ]:


data['Age'].fillna(data['Age'].median(), inplace=True)


# In[ ]:


# その他の特徴量エンジニアリングの部分の処理

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# ## 機械学習アルゴリズム
# 
# 機械学習アルゴリズムの大半は乱数を利用するので、再現性を担保するためにはseedを設定しておかなければなりません。実は2つ目のKernelを振り返ると、機械学習アルゴリズムのロジスティック回帰のハイパーパラメータとして random_state=0 を与え、seedを固定していました。

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)


# In[ ]:

