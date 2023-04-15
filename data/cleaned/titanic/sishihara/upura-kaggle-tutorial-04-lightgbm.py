#!/usr/bin/env python
# coding: utf-8

# This Notebook is a kaggle tutorial for Japanese kaggle beginners writen in Japanese.
# 
# # 4. 勾配ブースティングが最強？！ いろいろな機械学習アルゴリズムを使ってみよう

# これまでは機械学習アルゴリズムとして、ロジスティック回帰を採用していました。
# 
# この[Notebook](https://www.kaggle.com/sishihara/upura-kaggle-tutorial-04-lightgbm)では、いろいろな機械学習アルゴリズムを使ってみましょう。これまでロジスティック回帰を使っていた部分を差し替えて学習・予測を実行してみたいと思います。
# 
# ロジスティック回帰の実装に利用していたsklearnというパッケージは入出力のインタフェースが統一されており、手軽に機械学習アルゴリズムを変更できます。実際にいくつか試してみましょう。
# 
# また最近のKaggleのコンペティションで上位陣が利用している機械学習アルゴリズムとしては、勾配ブースティングやニューラルネットワークが挙げられます。これらはロジスティック回帰に比べて表現力が高く、高性能に予測できる可能性を秘めています。特に上位陣での採用率が高いのは「LightGBM」という勾配ブースティングのパッケージです。sklearnと同様のインターフェイスも用意されていますが、ここでは[Python-package Introduction](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)に記載の方式で実装します。

# In[ ]:


# 特徴量の準備

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
data['Age'].fillna(data['Age'].median(), inplace=True)
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1


# In[ ]:


data.head()


# In[ ]:


delete_columns = ['Name', 'PassengerId','Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)


# In[ ]:


X_train.head()


# 特徴量の準備が完了しました。
# 
# # sklearn
# まずはsklearn内で機械学習アルゴリズムを変更していきましょう。これまではロジスティック回帰を使ってきました。

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)


# sklearnでは、clfで宣言するモデルを切り替えるだけで機械学習アルゴリズムを差し替えられます。例えば、[ランダムフォレスト](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)と呼ばれる機械学習アルゴリズムを使ってみましょう。

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)


# あとはロジスティック回帰の場合と同様に学習・予測が実行可能です。

# In[ ]:

