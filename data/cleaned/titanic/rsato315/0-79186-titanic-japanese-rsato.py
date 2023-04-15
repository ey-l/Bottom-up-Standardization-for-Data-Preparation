#!/usr/bin/env python
# coding: utf-8

# ### データに必要なもののインポート
# - numpy, pandasのインポート, データの読み込み，テストデータ，学習用データの確認
# - データを用意に扱うためにライブラリをインポートする．
# - データを表示し，内容や欠損値,データの型を確認する．
# - これによりデータの解析に必要な情報を得る．

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
train_df = pd.read_csv("data/input/titanic/train.csv")
test_df = pd.read_csv("data/input/titanic/test.csv") 
train_df.head()


# In[ ]:


train_df.dtypes


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# ### 分析結果
# - 12個の変数を持ち，そのうち"Survived"は被説明変数，"PassengerId"はindexに分類されるため，それ以外の10個の説明変数から構成される．
# - Dtypeがobjectのものは文字列であり，そのままでは扱えないため，何らかの数値に変換することで，データの比較ができるようにする必要があることがわかる．
# - 学習データの891名と，テストデータの418名のデータのうち，"Age"と"Cabin","Fare", "Embarked"は全員分のデータが無い事がわかるため，予測への影響を少なくするために，これらのデータは平均値等で埋める必要がある．

# In[ ]:


train_df.info()


# ### データの確認
# - 学習データとテストデータを統合して一度に作業できるようにする．
# - ダミー変数で置き換えるため，データの種類を確認する．
# - データから特徴を探る．

# In[ ]:


# 統合データ
all_df = pd.concat([train_df, test_df], sort = False)
# インデックスを振り直す
all_df.reset_index( drop=True )


# In[ ]:


# 'Sex'の要素の種類を確認
all_df["Sex"].value_counts()


# In[ ]:


# 'Embarked'の要素の種類を確認
all_df["Embarked"].value_counts()


# In[ ]:


# 'Name'の要素の種類を確認
all_df["Name"].value_counts()


# In[ ]:


# 'Ticket'の要素の種類を確認
all_df["Ticket"].value_counts()


# In[ ]:


# データが固有のものかを調べる．
all_df.describe(include=['O'])


# In[ ]:


# 'Pclass'の要素の種類を確認
all_df["Pclass"].value_counts()


# In[ ]:


# 'SibSp'の要素の種類を確認
all_df["SibSp"].value_counts()


# In[ ]:


# 'SibSp'の要素が多いものの確認
all_df[all_df["SibSp"] > 4]


# In[ ]:


# 'Parch'の要素の種類を確認
all_df["Parch"].value_counts()


# In[ ]:


# 'Parch'の要素が多いものの確認
all_df[all_df["Parch"] > 4]


# In[ ]:


# 'Cabin'の要素の種類を確認
all_df["Cabin"].value_counts()


# In[ ]:


# 同一名のデータの確認1
all_df[all_df["Name"] == 'Kelly, Mr. James']


# In[ ]:


# 同一名のデータの確認2
all_df[all_df["Name"] == 'Connolly, Miss. Kate']


# In[ ]:


# 同一キャビンの確認1
all_df[all_df["Cabin"] == 'C23 C25 C27']


# In[ ]:


# 同一キャビンの確認2
all_df[all_df["Cabin"] == 'G6']


# In[ ]:


# 同一キャビンの確認3
all_df[all_df["Cabin"] == 'B57 B59 B63 B66']


# In[ ]:


# 'Cabin'の頭文字の要素ごとの個数
Cabin_str = all_df["Cabin"].str[0]
Cabin_str.value_counts()


# ## データからわかったこと
# 
# ### 'Sex'
# - 欠損データがない．男女比は約２：１
# 
# ### 'Pclass'
# - 欠損データは６つ．
# - データは1,2,3の３つから構成され，3が最も多い．
# - 'Pclass'は'Passenger Class'の略で座席のクラスを表すと考えられ，3が最も多いことから３が一番低いクラスであると考えられる．
# - これは，乗客の経済的な地位を表していると考えられ，生死に大きく影響しているとみられる．
# - 欠損値は最も数の多い3で埋めることとする．
# 
# ### 'Embarked'
# - テストデータで2つ欠損．S，C，Qの3種類のみから構成され，全体の約7割が'S'である事がわかった．したがって，テストデータの欠損値は最も頻度の多い'S'で補間すればよいとわかる．
# - S,C,Qは，他の人のDiscussionから乗船位置を示していることがわかった．したがって，船舶内での滞在位置などにも大きく関わっていると考えられ生存の可否に大きく関わるデータと予想される
# 
# ### 'Name'
# - 同一名が2組含まれていた．
# - これらの情報を詳しく見比べた結果，両者は，データが何らかの理由により同一人物が重複して記録されたわけではなく，同一名の別人であることがわかった．
# - 名前が同じかどうかは，生存の可否には一般的に関係がないと考えられるため，学習時に'Name'属性をデータから外すか，ダミー変数で両者を別の人として区別する必要がある．
# - あるいは，後述する'SibSp'にて同一のファミリーネームが複数組確認されたため，家族ごとに生存の可否を予測することも可能かもしれない．
# 
# ### 'Ticket'
# - 数字のみで構成されているもの，
# - CAやSTONなどの文字列とその後ろに数字の並ぶ構成のものがある．
# - これは，uniqueの個数が929/1309(71%)であり，ほとんど固有の要素をもつ属性と考えられるため，学習データからは外すべきと考えられる．
# 
# ### 'SibSp'
# - 欠損値はなし．0から8で構成され，0，続いて1が最も多い．
# - 5以上の要素について確認した結果，SibSp==5は全て，'Name'属性が全て'GoodWin'であり，SibSp==8は全て'Sage'だったため，これらは全てファミリーネームである
# - SibSpの意味は，Siblings/Spouses(親戚，配偶者)を指す略語だと考えられる．
# - これらの家族は，学習データ上は誰も生存していない．
# - したがって，家族ごとに生存の可否を学習できる可能性があり，そのためのデータとして，'SibSp'属性は参考になる可能性がある．
# 
# ### 'Parch'
# - 欠損値なし．0がほとんどを占め，1,2..と数字が多くなるほどその値を持つ人は少なくなる．
# - 'Parch'の5以上の要素について確認した結果，'SibSp'と同じデータも検出され，40歳前後であった．
# - この'Parch'はParents/Children（親子）の数であると考えられるため，前述の家族ごとのデータとしての検討に活用できそうである．
# 
# ### 'Cabin'
# - 欠損値が，1014/1309と非常に多く，ユニークな値が196/295であるため，生存の可否をこのまま予測するのは難しいと考えられる．
# - しかし，頭文字のアルファベットが，8ケースしかなかったため，キャビンの位置等で，生存率の予測に活用できる可能性がある．D->B->Cの順に大きくなる．

# ## 性別のより詳しい属性の作成
# - '.'で分離することで，'Mr','Mrs','Miss','Master'などの敬称("Honorific"属性)を家族構成としての学習データに利用できる．
# - 前述の略称以外の少ない頻度の要素と欠損値の場合でも，'Parch'と，'Age'，'Sex'からどれかに当てはめる事ができる．
# - 今回は，学習への影響を減らすため，女性と男性で最も多い'Miss'と'Mr'でそれぞれ補間することとする
# - なお，これらの処理により，'Sex'属性のデータは"Honorific"属性で全て置き換えられたこととなるため，学習時には省く．

# In[ ]:


# 文字列を','と'.'で分割する
name = all_df["Name"].str.split( "[, .]", expand=True )
name


# In[ ]:


# name変数の2つ目の要素を取り出す
all_df["Honorific"] = name[2]
all_df["Honorific"].value_counts()


# In[ ]:


#'Mr', 'Mrs', 'Miss', 'Master'以外の要素を'X'と変換する
name = all_df["Honorific"].copy()
name[::] = 'X'
name[ all_df["Honorific"] == "Mr" ] = "Mr"
name[ all_df["Honorific"] == "Mrs" ] = "Mrs"
name[ all_df["Honorific"] == "Miss" ] = "Miss"
name[ all_df["Honorific"] == "Master" ] = "Master"
all_df["Honorific"] = name
name


# In[ ]:


# Xの要素を'Sex'が'male'なら'Mr', 'female'なら'Miss'に置き換える
all_df.loc[(all_df["Honorific"] == 'X') & (all_df['Sex'] == 'male'),"Honorific"] = 'Mr'
all_df.loc[(all_df["Honorific"] == 'X') & (all_df['Sex'] == 'female'),"Honorific"] = 'Miss'
all_df["Honorific"].value_counts()


# ## 結果
# - 全ての'Name'属性が４つのいずれかに分類できた．

# ## 'Familiy'属性の作成
# - 前述の結果から，'Name'属性のファミリーネームを分離することで，'Family'属性を新しく作ることで新しいデータを得られると考えられる．
# - ファミリーネームを分離するために，','で切り取ったname変数のインデックス１を新しい属性として割り当てる．

# In[ ]:


# 文字列を','で分割する
name = all_df["Name"].str.split( "[, ]", expand=True )
name


# In[ ]:


# ファミリーネームを確認
name[0].value_counts()


# ## 結果
# - ファミリーネームは種類が866もあり，ほとんどユニークな値であることがわかったため，'Family'属性をデータとして扱うことは，やめることとした．

# ## 'Cabin'のデータを整形する
# - 欠損値が非常に多いため，欠損値を'Another'とし，データの少ない'G','T'も含める．
# - データの中には，複数の座席が記録されているものがあるが，アルファベットは同一の場合が多いため，頭文字のみ適用する

# In[ ]:


# 頭文字だけ取り出したものを，"Cabin_ini"属性に入れる．
all_df["Cabin_ini"] = all_df["Cabin"].str[0]
all_df["Cabin_ini"]


# In[ ]:


# 'G','T'をNanに置き換える
all_df["Cabin_ini"].mask((all_df["Cabin_ini"] == 'G') | (all_df["Cabin_ini"] == 'T'), 'Another', inplace=True)

# NaNを埋め,要素を確認する
all_df.fillna(value={"Cabin_ini":'Another'}, inplace=True)
all_df["Cabin_ini"].value_counts()


# ## 結果
# - 欠損値が非常に多いため，これを学習データに利用できない可能性が高い

# ## 欠損値の補間
# - 'Age'と'Fare'の欠損値は平均値で埋めることで，影響を最小限にする．
# - 'Pclass'の欠損値は3, 'Embarked'の欠損値は'S'で補間する．いずれも最頻値である．

# In[ ]:


all_df["Age"].fillna( all_df["Age"].mean() , inplace=True) 
all_df["Fare"].fillna( all_df["Fare"].mean(), inplace=True)
all_df["Pclass"].fillna( 3, inplace=True)
all_df["Embarked"].fillna( 'S', inplace=True)
# 欠損値の確認
all_df.isna().sum()


# In[ ]:


all_df_re = all_df.drop([ "Cabin", "Name", "Sex", "Ticket"],axis=1)
# ダミー変数への変換
all_df_re = pd.get_dummies(all_df_re,drop_first=True)


# ## 結果
# - 欠損値を埋めて，ダミー変数への変換が完了した．
# - CabinはCabin_iniを代用するためそのままとする．

# In[ ]:


# 再び，trainとtestに戻す
train_df_re = all_df_re[:len(train_df)]
test_df_re = all_df_re[len(train_df):]
#説明変数と目的変数の分離 (列を削除) 
train_X = train_df_re.drop( ["Survived", "PassengerId"],axis=1 )
test_X = test_df_re.drop( ["Survived", "PassengerId"], axis=1 )
train_Y = train_df_re["Survived"]
train_df_re


# ## 説明変数から除外するもの
# - 前述のとおり，以下のものを除外する
#     "PassengerId", "Cabin", "Name", "Sex", "Ticket"
# - 'Sex'は'Honorific'で代用する．
# - 'Cabin'は'Cabin_ini'で代用する．

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# 固定のSEED値を設定
SEED = 7017
model = RandomForestClassifier( n_estimators=1500, max_depth=500, random_state=SEED )