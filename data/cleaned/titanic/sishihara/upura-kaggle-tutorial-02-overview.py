#!/usr/bin/env python
# coding: utf-8

# This Notebook is a kaggle tutorial for Japanese kaggle beginners writen in Japanese.
# 
# # 2. 全体像を把握！ submitまでの処理の流れを見てみよう

# この[Notebook](https://www.kaggle.com/sishihara/upura-kaggle-tutorial-02-overview)では、前回は一旦無視したNotebookの処理の流れを具体的に見ていきます。
# 
# ぜひ、実際に一番上からセルを実行しながら読み進めてみてください。
# 
# 具体的な処理の流れは、次のようになっています。
# 
# 1. パッケージの読み込み
# 2. データの読み込み
# 3. 特徴量エンジニアリング
# 4. 機械学習アルゴリズムの学習・予測
# 5. submit（提出）
# 

# ## パッケージの読み込み
# ここでは、以降の処理で利用する「パッケージ」をimportします。
# パッケージをimportすることで、標準では搭載されていない便利な機能を拡張して利用できます。
# 
# 例えば次のセルでimportするnumpyは数値計算に秀でたパッケージで、pandasはTitanicのようなテーブル形式のデータを扱いやすいパッケージです。
# 
# ここでは、最初に必要な2つのパッケージをimportしています。importはどこで実施しても構いません。（特にScript形式の場合は、冒頭でのimportが望ましいです）

# In[ ]:


import numpy as np
import pandas as pd


# ## データの読み込み
# 
# ここでは、Kaggleから提供されたデータを読み込みます。
# 
# まずはどういうデータが用意されているかを確認しましょう。詳細は[「Data」タブ](https://www.kaggle.com/c/titanic/data)に記載されています。

# In[ ]:





# In[ ]:


train = pd.read_csv("data/input/titanic/train.csv")
test = pd.read_csv("data/input/titanic/test.csv")
gender_submission = pd.read_csv("data/input/titanic/gender_submission.csv")


# 「gender_submission.csv」は、submitのサンプルです。このファイルで提出ファイルの形式を確認できます。仮の予測として、女性のみが生存する（Survivedが1）という値が設定されています。

# In[ ]:


gender_submission.head()


# 「train.csv」は機械学習の訓練用のデータです。これらのデータについてはTitanic号の乗客の性別・年齢などの属性情報と、その乗客に対応する生存したか否かの情報（Survived）が格納されています。

# In[ ]:


train.head()


# 「test.csv」は、予測を実施するデータです。これらのデータについてはTitanic号の乗客の性別・年齢などの属性情報のみが格納されており、訓練用データの情報を基に予測値を算出することになります。
# 
# 「train.csv」と比較すると、Survivedという列が存在しないと分かります。（この列があったら全て正解できてしまうので当然ですね）

# In[ ]:


test.head()


# これらは、Kaggleから提供された大元のデータです。
# 
# 例えばName, Sexなどは文字列で格納されており、そのままでは機械学習アルゴリズムの入力にすることはできません。
# 機械学習アルゴリズムが扱える数値の形式に変換していく必要があります。
# 
# NaNというのは、データの欠損です。こうした欠損値は、一部の機械学習アルゴリズムではそのまま扱うこともできますが、平均値など代表的な値で穴埋めする場合も多いです。
# 
# こういった処理を「特徴量エンジニアリング」と呼びます。
# 特徴量エンジニアリングに当たって、Kaggleでは「train.csv」と「test.csv」をまとめて扱う方が都合が良いので、dataという形でこの段階で結合しておきます。

# In[ ]:


data = pd.concat([train, test], sort=False)


# In[ ]:


data.head()


# 当たり前ですが (trainのデータ数) + (testのデータ数) == (dataのデータ数) になっています。

# In[ ]:


print(len(train), len(test), len(data))


# 欠損値の有無を確認すると、次のようになっていました。

# In[ ]:


data.isnull().sum()


# ## 特徴量エンジニアリング

# ### 1. Pclass
# 
# Pclassは、チケットの階級です。生々しい話ですが、VIPルームの乗客の方が優先的に救出される可能性があるなど、予測に寄与する可能性があります。
# 
# データに欠損はなく、数値として格納されています。機械学習アルゴリズムに入力するに当たって、特に加工は必要ありません。

# In[ ]:


data['Pclass'].value_counts()


# ### 2. Sex
# 
# Sexは、性別です。緊急時、腕力のある男性は他人の救護に関わるなどが考えられ、予測に寄与する可能性がありそうです。
# 
# データに欠損はないですが、文字列として格納されているので、0, 1の数値に変換しておきます。

# In[ ]:


data['Sex'].replace(['male','female'], [0, 1], inplace=True)


# ### 3. Embarked
# 
# Embarkedは、出港地（どこでTitanicに乗ったか）を示します。
# 
# >C - Cherbourg, S - Southampton, Q = Queenstown
# 
# 内訳を見ると、Sが多数を占めています。欠損が2個あるので、Sで埋めておきましょう。その後にSexと同様、数値に変換しておきます。

# In[ ]:


data['Embarked'].value_counts()


# In[ ]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# ### 4. Fare
# 
# Fareは運賃で、Pclassと同様に予測に寄与する可能性があります。欠損値が1個あるので、平均値で穴埋めしておきましょう。

# In[ ]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)


# ### 5. Age
# 
# Ageは年齢です。子供の方が優先的に救助されるなど、予測に寄与する可能性がありそうです。
# 
# 欠損は263個あります。単純に平均値で埋めても良いですが、少しトリッキーに標準偏差を考慮した乱数で穴埋めしてみました。

# In[ ]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)


# まだまだ列はありますが、今回はその他の列は一旦無視して話を進めましょう。

# In[ ]:


delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)


# ここでdataを結合元のtrain, testに戻しておきます。

# In[ ]:


train = data[:len(train)]
test = data[len(train):]


# 最後に、trainを特徴量部分と予測の対象に分割しておきましょう。

# In[ ]:


y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# ## 機械学習アルゴリズム
# 
# いよいよ、用意した特徴量と予測の対象のペアから、機械学習アルゴリズムを用いて予測器を学習させましょう。
# 
# ここではロジスティック回帰という機械学習アルゴリズムを利用します。

# In[ ]:


from sklearn.linear_model import LogisticRegression


# 予測器を宣言します。括弧内の値はハイパーパラメータと呼ばれ、予測器の振る舞いを決める要素です。（ここでは適当に設定していますが、5つ目のKernelでは調整方法を学びます）

# In[ ]:


clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)


# clfに、特徴量と予測の対象のペアを渡してfitさせることで、学習が進みます。

# In[ ]:

