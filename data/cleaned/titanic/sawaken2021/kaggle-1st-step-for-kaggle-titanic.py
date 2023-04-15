#!/usr/bin/env python
# coding: utf-8

# ## はじめに (Introduction)
# 多くの日本人の方が、Kaggleを通した機械学習の学び・遊びにおいて英語が壁になっているようなので、日本語・英語併記で書いていきたいと思います。ひとりでも多くの日本人Kagglerが誕生する事を願っています。
# 
# <span style="color:#009900;">Many Japanese people seem to find some difficulties to study/play machine learning through Kaggle. So I'd like to write this note in both Japanese and English. I hope we could have many Japanese Kagglers in the near future.</span>
# 
# このKernelでは、Kaggleに初めて挑戦する方向けに、Pythonでコードを書いて、最初のSubmitをするまでを説明します。
# 
# <span style="color:#009900;">This kernel explains how to write a Python code and to make your 1st submit for Kaggle beginners.</span>
# 
# データを取り扱うのに便利な[pandas](https://ja.wikipedia.org/wiki/Pandas)ライブラリと、データの可視化に便利な[matplotlib](https://ja.wikipedia.org/wiki/Matplotlib)ライブラリの使い方の基礎も紹介しています。
# 
# <span style="color:#009900;">Some very basic usage of [pandas](https://en.wikipedia.org/wiki/Pandas) library which is useful for handling data and [matplotlib](https://en.wikipedia.org/wiki/Matplotlib) library which is useful for visualizing data are also introduced.</span>

# ## pandasとmatplotlibのインストール (Installation of pandas and matplotlib)
# もしまだpandasライブラリとmatplotlibライブラリをインストールしていない場合は、インストールしておきましょう。インストール方法は使用中の環境によって異なりますが、Pythonを直接使用している場合は以下のコマンドでインストールできます。Anaconda等の他の環境の場合はGoogle等で検索してみて下さい。すぐにやり方が見つかると思います。
# 
# <span style="color:#009900;">If you have not yet installed pandas library and matplotlib library, please install them at this point. The install commands are different depending on your environment, but if you are directly using Python, you can use the following commands. In case of Anaconda or anything like that, please try to google it. I believe you can find the necessary information soon.</span>
# 
# ~~~python
# pip install pandas
# pip install matplotlib
# ~~~

# ## ライブラリのインポート (Import Libraries)
# 
# まず最初に、Pythonコードの先頭に使用するライブラリのインポート宣言を書きましょう。今回はpandasライブラリとmatplotlibライブラリを使うので、以下のようなコードになります。
# 
# <span style="color:#009900;">At first, let's write some declarations of importing libraries used in this code at the very beginning of the Python file. This time, we are using pandas library and matplotlib library, so the code will be like below.</span>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# これ以降、pandasのライブラリは**pd**、matplotlibのライブラリは**plt**でアクセスする事が出来ます。
# 
# <span style="color:#009900;">After this, you can access pandas library via **pd**, and matplotlib library via **plt**.</span>

# ## 訓練データの読み込み (Reading Training Data)
# 
# タイタニックではtrain.csvファイルとtest.csvファイルという2つのファイルが用意されています。train.csvには乗客の生死の結果が記述されており、このデータを元に機械学習していく事になります。一方、test.csvには乗客の生死が書かれておらず、このtest.csvの乗客の生死を予測してその精度を競い合う事になります。
# 
# <span style="color:#009900;">In Titanic, you have 2 CSV files named train.csv and test.csv. In train.csv, the result of passengers' status (Dead or Survived) is described, and you will let your computer learn based on this data. On the other hand, you have no data about passengers' status in test.csv. You will predict the passengers' status and will compete the accuracy of the prediction with Kagglers.</span>
# 
# まずは、訓練データを読み込みましょう。
# 
# <span style="color:#009900;">Let's get started from reading training data.</span>

# In[ ]:


train_df = pd.read_csv('data/input/titanic/train.csv', header=0)


# `pd.read_csv()`の2番目の引数`header=0`は、0行目、つまり最初の行をヘッダとして読み込みなさいという意味です。train.csvを見ると、最初の行はデータの意味を示すヘッダ情報が書かれていることがわかります。
# 
# <span style="color:#009900;">The `pd.read_csv()`'s 2nd argument `header=0` means "recognize the first line as header". When you open and take a look at train.csv, you can find header information describing the meaning of each data in the first row.</span>

# ## 欠損データ (Missing Data)
# これからデータ分析に入りますが、その前に欠損データについて少し説明しておきます。このタイタニックの訓練データは全部で891名分のデータがあり、氏名、性別、年齢、客室クラスなど様々な情報が記載されています。しかし、必ずしも全てのデータが揃っているわけではありません。例えば年齢については117名分のデータは空白となっています。こうしたデータを欠損データと呼びます。欠損データをどのように扱うか、どのように補完するかはデータ分析する人の腕の見せ所でもありますが、ここでは欠損データが一切無い性別データを見ていきたいと思います。
# 
# <span style="color:#009900;">We're going into data analysis, but before that, let's talk a little bit about missing data. This Titanic's training data has data for 891 passengers in total. You have various information such as name, gender, age, cabin class etc. However, not all data is fully prepared. For example, 117 passengers' age data is blank. Such data is called missing data. It's good time to show off your skills how to handle and complement these missing data, however we'd like to focus on gender data here which has no missing data.</span>

# ## 男女の数 (Number of Male/Female)
# 「Sex」列に「female」とあるのが女性、「male」とあるのが男性です。`train_df.query("Sex == 'female'")`の結果をデータセットfamaleに入力することで女性のみのデータセットfemaleが出来上がります。同様に男性用のデータセットを2行目で生成しています。
# 
# <span style="color:#009900;">In the "Sex" column, "female" is a female, and "male" is a male. By setting `train_df.query("Sex == 'female'")` into the dataset female, you can create a dataset including females only. Similarly, a dataset for males is generated in the 2nd line.</span>
# 
# このデータセットの列数は`len()`関数で得られるので、女性の数は`len(female)`、男性の数は`len(male)`でわかります。`print()`で出力する際は、`str()`を用いて整数を文字列に変換してから出力しています。
# 
# <span style="color:#009900;">The number of rows in this dataset can be obtained with `len()` function, so the number of females can be calculated by `len(female)` and the number of males can be calculated by `len(male)`. When outputting by `print()`, you need to convert integer value to a string using `str()`.</span>

# In[ ]:


female = train_df.query("Sex == 'female'")
male = train_df.query("Sex == 'male'")

print('Number of Female = ' + str(len(female)))
print('Number of Male   = ' + str(len(male)))


# ## 男女の数の視覚化 (Visualization of Number of Male/Female)
# 画面上に表示された数字だけでは寂しいので、matplotlibライブラリを用いてグラフにプロットしてみましょう。
# 
# <span style="color:#009900;">Displaying numbers only is a bit too simple, so let's plot in a graph using matplotlib library.</span>
# 
# まずは棒グラフにしてみます。`plt.title()`でグラフのタイトルを、`plt.bar()`では以下の設定をしています。
# 
# <span style="color:#009900;">Start with a bar chart. The graph title is set by `plt.title()`, and the following properties are set by `plt.bar()`.</span>
# 
# 1. X軸の値(Female、Male) <span style="color:#009900;">X-Axis value (Female, Male)</span>
# 1. Y軸の値(女性の数、男性の数) <span style="color:#009900;">Y-Axis value (Number of female, Number of male)</span>
# 1. 棒グラフの色(淡い赤、淡い青) <span style="color:#009900;">Bar chart color (Pale red, Pale blue)</span>
# 

# 


# In[ ]:


plt.title('Female vs Male (Bar Graph)')
plt.bar(['Female', 'Male'], [len(female), len(male)], color=['mistyrose', 'lightblue'])



# グラフにしてみましたが、逆に数値が無くなってしまいました。棒グラフの上に数値を追加しましょう。具体的には`plt.text()`を用いてグラフ上に文字列を追加します。
# 
# <span style="color:#009900;">We made a graph, but conversely, the numbers have disappeared. Let's add numbers on the bar chart. Specifically, add strings above the graph using `plt.text()`.</span>
# 
# `plt.text()`の1番目の引数はX軸の位置、2番目の引数はY軸の位置になります。3番目の引数は表示する値です。`ha='center'`は表示する値のセンタリング、`va='bottom'`は表示する値の縦方向の位置を棒グラフの上に表示されるように調整しています(この指定が無いと文字列と棒グラフの上部が重なって表示されていまいます)。
# 
# <span style="color:#009900;">The first argument of `plt.text()` is the X-axis position, and the second argument is the Y-axis position. The third argument is the value to display. `ha = 'center'` centers the displayed value, and `va ='bottom'` adjusts the vertical position of the displayed value to be located above the bar graph (if not specified, the text and the top of the bar chart will be overlapped).</span>

# In[ ]:


plt.title('Female vs Male (Bar Graph w/ Label)')
plt.bar(['Female', 'Male'], [len(female), len(male)], color=['mistyrose', 'lightblue'])
plt.text('Female', len(female), len(female), ha='center', va='bottom')
plt.text('Male', len(male), len(male), ha='center', va='bottom')



# こうしたグラフを作る際は、少し淡い色を使った方が見た目が良い感じになります。redやblueなどの代表色だとかなりどぎついグラフになってしまいます。
# 
# <span style="color:#009900;">When making graphs, it may look better to use slightly lighter colors. If using representative colors such as red or blue, your graph will be an irritating one.</span>

# ## 男女別の死亡・生存数 (Number of Deaths/Survivors by Gender)
# 
# 男女の総数がわかったら、この中で何人が死亡し、何人が生き残ったかを見ていくことにしましょう。pandasライブラリは非常に強力なライブラリで、データ抽出の方法はいろいろあるのですが、ここでは最初に男女別のデータ数を計算した時と同じやり方で算出してみる事にします。
# 
# <span style="color:#009900;">Once you know the total number of males and females, let's see how many people died or survived. The pandas library is a very powerful library, so that there are several ways to extract data. However here we will try to extract them in the same way as when we first extracted gender-specific data.</span>
# 
# 4つのデータセットを用意しました。
# 
# <span style="color:#009900;">4 types of list are prepared.</span>
# 
# | リスト (List) | 内容 (Description) |
# |---|---| 
# |female_dead | 死亡した女性 (Dead females) |
# |female_survived | 生存した女性 (Survived females) |
# |male_dead | 死亡した男性 (Dead males) |
# |male_survived | 生存した男性 (Survived males) |
# 
# `female_dead`を例に説明すると、死亡した女性は「Sex」列が「female」かつ「Survived」列が0(死亡)のデータになりますので、以下の式でデータ抽出する事が出来ます。
# 
# <span style="color:#009900;">In the case of `female_dead`, died females belong to the data where “Sex” column is "female" and “Survived” column is 0 (death). So the data can be extracted using the following script.</span>
# 
# `female_dead = train_df.query("Sex == 'female' & Survived == 0")`
# 
# ANDの論理記号を`&&`とかにしたりすると、正常に動作しないので気を付けて下さい。
# 
# <span style="color:#009900;">Note that if you change the AND logical symbol to `&&`, it will not work properly.</span>

# In[ ]:


female_dead = train_df.query("Sex == 'female' & Survived == 0")
female_survived = train_df.query("Sex == 'female' & Survived == 1")
male_dead = train_df.query("Sex == 'male' & Survived == 0")
male_survived = train_df.query("Sex == 'male' & Survived == 1")

print('Number of Female (Dead)     = ' + str(len(female_dead)))
print('Number of Female (Survived) = ' + str(len(female_survived)))
print('Number of Male (Dead)       = ' + str(len(male_dead)))
print('Number of Male (Survived)   = ' + str(len(male_survived)))


# これも同じようにグラフで視覚化してみましょう。今回は死者と生存者の2種類のデータがありますので、積み上げ棒グラフで表示してみます。
# 
# <span style="color:#009900;">Let's visualize these data in the same way in a graph. We have two types of data (dead and survivors), so let's show them in a stacked bar chart.</span>
# 
# 積み上げ棒グラフの作り方は、2つの棒グラフを描画するだけで、それほど難しいものではありません。最初に死者の棒グラフを通常の棒グラフと同じ方法で作成し、続いて生存者の棒グラフを`bottom=[len(female_dead), len(male_dead)]`属性付きで作成します。`bottom`属性によって、棒グラフの底(始点)を最初に描画した死者の棒グラフの一番上に指定することで、積み上げ棒グラフになります。
# 
# <span style="color:#009900;">The way to make a stacked bar chart is not so difficult. All you need to do is just making two bar charts. First, create a bar chart of the dead in the same way as a usual bar chart, and then create a bar chart of the survivors with `bottom = [len(female_dead), len(male_dead)]`. This attribute specifies the bottom (starting point) of the bar chart at the top of the bar chart of the dead, which results in a stacked bar chart.</span>

# In[ ]:


plt.title('Female vs Male (Stacked Bar Graph)')
plt.bar(['Female', 'Male'], [len(female_dead), len(male_dead)], color=['darkred', 'midnightblue'])
plt.bar(['Female', 'Male'], [len(female_survived), len(male_survived)], bottom=[len(female_dead), len(male_dead)], color=['mistyrose', 'lightblue'])



# 例によってデータ値が無くわかりづらいグラフになってしまいました。少し煩雑ですが、死者・生存者の数を男女別にグラフの上に重ねて表示させます。
# 
# <span style="color:#009900;">As usual, data value has disappeared and it has become a hard-to-understand graph. Although it's a little complicated procedure, however let's try to display the number of deaths / survivors on the graph by gender.</span>
# 
# 鍵となるのは、Y軸の座標位置を指定する`plt.text()`の2番目の引数です。棒グラフの中央に重ねて表示させるため、棒グラフの下部に棒グラフの長さの半分を加えた値を指定しています。また`va='center'`を指定することで、文字列が縦方向にセンタリングされ、棒グラフの中央に表示されるよう微調整されます。
# 
# <span style="color:#009900;">The key is the second argument of `plt.text()`, which specifies the Y-axis position. We're setting this 2nd argument with the bottom of the bar plus half the length of the bar in order to show the data value in the middle of the bar. By specifying `va='center'`, the text strings position will be fine-adjusted to the middle of the bar.</span>
# 
# | データ種別 (Data Type) | Y軸の位置 (Y-Axis Position) |
# |---|---| 
# |死亡した女性 | len(female_dead)/2 |
# |生存した女性 | len(female_dead) + len(female_survived)/2 |
# |死亡した男性 | len(male_dead)/2 |
# |生存した男性 | len(male_dead) + len(male_survived)/2 |

# In[ ]:


plt.title('Female vs Male (Stacked Bar Graph w/ Label)')
plt.bar(['Female', 'Male'], [len(female_dead), len(male_dead)], color=['darkred', 'midnightblue'])
plt.bar(['Female', 'Male'], [len(female_survived), len(male_survived)], bottom=[len(female_dead), len(male_dead)], color=['mistyrose', 'lightblue'])
plt.text('Female', len(female_dead)/2, "Dead\n" + str(len(female_dead)), color='white', ha='center', va='center')
plt.text('Female', len(female_dead) + len(male_survived)/2, "Survived\n" + str(len(male_survived)), color='black', ha='center', va='center')
plt.text('Male', len(male_dead)/2, "Dead\n" + str(len(male_dead)), color='white', ha='center', va='center')
plt.text('Male', len(male_dead) + len(male_survived)/2, "Survived\n" + str(len(male_survived)), color='black', ha='center', va='center')



# これを見てわかるのは、圧倒的な女性優位の状況です。女性は74.2%が生き残った一方で、男性の生存率は実に18.8%しかありません。
# 
# <span style="color:#009900;">You can easily find the overwhelming female-advantaged situation. While 74.2% of females survived, the survival rate of males is only 18.8%.</span>
# 
# タイタニックの死者がこれほどまで膨らんだ要因のひとつに救命ボートの少なさがありました。このデータは女性が優先的に救命ボートへ誘導された事が伺えます。
# 
# <span style="color:#009900;">One of the factors that caused Titanic's tragedy was the shortage of lifeboats. This data shows that females were preferentially guided to the lifeboat.</span>

# ## 機械学習を利用しない予測 (Prediction w/o Machine Learning)
# 
# こうなると、最もシンプルな死者・生存者の予測モデルは、男性なら死亡、女性なら生存と予測してしまう事です。ここに機械学習の要素は全くありませんが、まずはこうしたスタティックな予測でどれだけ予測が当たるかを見てみましょう。その後、機械学習を利用した予測モデルの改良をする事で、いかに予測の精度が向上していくかを実感する事ができれば、機械学習の技術習得やKaggleがとても楽しいものになってくると思います。
# 
# <span style="color:#009900;">The simplest death-survivor prediction model would be predicting all males die and all females survive. There is no element of machine learning here, but let's take a look at how good or bad this kind of static prediction is. After that, by improving the prediction model using machine learning, when you see how the accuracy of prediction improves, I believe that studying of machine learning techniques in Kaggle will be very enjoyable for you.</span>
# 
# 訓練データ同様、予測すべきデータが格納されている試験データを読み込みます。
# 
# <span style="color:#009900;">As with training data, read test data containing data to be predicted.</span>

# In[ ]:


test_df = pd.read_csv('data/input/titanic/test.csv', header=0)


# 「Survived」列を追加し0で初期化します。そして、「Sex」列を見て「female」だったら「Survived」を1に設定します。
# 
# <span style="color:#009900;">Add "Survived" column and initialize it with 0. Then, if "Sex" column is "female", set "Survived" to 1.</span>

# In[ ]:


test_df["Survived"] = 0
test_df.loc[test_df["Sex"] == 'female', "Survived"] = 1


# 元々存在していた「PasserId」列と、先ほど追加した「Survived」列以外は不要なので、`drop()`メソッドを用いて削除します。最後の`axis=1`は削除の対象が行ではなく列である事を意味しています。
# 
# <span style="color:#009900;">Except for the originally existing "PasserId" column and the "Survived" column we have added before, remove the rest of columns by using `drop()` method. The last argument `axis = 1` means that the target of deletion is not a row but a column.</span>

# In[ ]:


test_df = test_df.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)


# 念のため、test_dfの最初の5行と最後の5行を表示してみましょう。
# 
# <span style="color:#009900;">Just in case, let's display the first five lines of test_df and the last five lines.</span>

# In[ ]:


print(test_df.head(5))
print(test_df.tail(5))



# 


# In[ ]:





# 最後に、Kaggle画面右上の「Submit Predictions」をクリックして出来上がったCSVファイルをドラッグ＆ドロップすれば、おめでとうございます！これであなたもKagglerの仲間入りです。
# 
# <span style="color:#009900;">Finally, click "Submit Predictions" in the upper right corner of the Kaggle screen, and drag and drop the generated CSV file. Congratulations! You are now a Kaggler!</span>

# ## 終わりに (At The End)
# 
# 一番最初のスコア(予測の的中率)はいくつでしたか？今回は機械学習無しの随分と粗削りな予測をしましたが、思いのほか高い精度で予測出来ていたのではないでしょうか。
# 
# <span style="color:#009900;">What was your first score (prediction score)? This time, we made rough prediction without machine learning, but the prediction accuracy was much higher than you thought, wasn't it?</span>
# 
# 次回はなんでもよいのでちゃんと機械学習を取り込んで、男性でも生き残れるようにしてみて下さい(笑) そしてその成果を是非Kaggle上で皆さんと共有して下さい。楽しみにしています。
# 
# <span style="color:#009900;">Next time please try to make use of machine learning, and to rescue men too :-) Then please share the results with us in Kaggle. Looking forward to your posts.</span>