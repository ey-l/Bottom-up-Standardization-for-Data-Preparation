import numpy as np
import pandas as pd
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud
from PIL import Image
import plotly.express as px
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
test_path = 'data/input/nlp-getting-started/test.csv'
test = pd.read_csv(test_path)
train_path = 'data/input/nlp-getting-started/train.csv'
train = pd.read_csv(train_path)
disaster = train[train['target'] == 1]
not_disaster = train[train['target'] == 0]
disaster.head()
not_disaster.head()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
nested_list_disaster = []
for i in disaster.text:
    words = word_tokenize(i)
    filtered_tweets = [stemmer.stem(word) for word in words if word is not stop_words and word.isalpha()]
    nested_list_disaster.append(filtered_tweets)
flatten = [item for sublist in nested_list_disaster for item in sublist]
unique_words_disaster = pd.Series(flatten).value_counts()
frequent_words_disaster = unique_words_disaster[:]
top_10_most_frequent_disaster = frequent_words_disaster[:10]
fig = px.bar(x=top_10_most_frequent_disaster.index, y=top_10_most_frequent_disaster.values, color=top_10_most_frequent_disaster.index, title='Most Frequent Top 10 : Disaster', labels=dict(x='Words', y='Count of Words'))
fig.show()
disaster_word_cloud = pd.DataFrame(flatten)
disaster_word_cloud.columns = ['Words']
title_words = pd.DataFrame(list(zip(disaster_word_cloud['Words'].value_counts().index, disaster_word_cloud['Words'].value_counts())), columns=['Words', 'Count'], index=None)
title_words = dict(zip(title_words['Words'].tolist(), title_words['Count'].tolist()))
wc = wordcloud.WordCloud(width=1200, height=500, collocations=False, relative_scaling=0, background_color='black', colormap='Reds').generate_from_frequencies(title_words)
plt.figure(figsize=(15, 5), dpi=300)
plt.title('Disaster WordCloud')
plt.imshow(wc, interpolation='bilinear', alpha=1)
_ = plt.axis('off')
nested_list_not_disaster = []
for i in not_disaster.text:
    words = word_tokenize(i)
    filtered_tweets = [stemmer.stem(word) for word in words if word is not stop_words and word.isalpha()]
    nested_list_not_disaster.append(filtered_tweets)
flatten = [item for sublist in nested_list_not_disaster for item in sublist]
unique_words_not_disaster = pd.Series(flatten).value_counts()
frequent_words_not_disaster = unique_words_not_disaster[:]
top_10_most_frequent_not_disaster = frequent_words_not_disaster[:10]
fig = px.bar(x=top_10_most_frequent_not_disaster.index, y=top_10_most_frequent_not_disaster.values, color=top_10_most_frequent_not_disaster.index, title='Most Frequent Top 10 : Not Disaster', labels=dict(x='Words', y='Count of Words'))
fig.show()
not_disaster_word_cloud = pd.DataFrame(flatten)
not_disaster_word_cloud.columns = ['Words']
title_words = pd.DataFrame(list(zip(not_disaster_word_cloud['Words'].value_counts().index, not_disaster_word_cloud['Words'].value_counts())), columns=['Words', 'Count'], index=None)
title_words = dict(zip(title_words['Words'].tolist(), title_words['Count'].tolist()))
wc = wordcloud.WordCloud(width=1200, height=500, collocations=False, relative_scaling=0, background_color='black', colormap='Blues').generate_from_frequencies(title_words)
plt.figure(figsize=(15, 5), dpi=300)
plt.title('Not Disaster WordCloud')
plt.imshow(wc, interpolation='bilinear', alpha=1)
_ = plt.axis('off')
word_size = 12142
nested_list = []
for i in train.text:
    words = word_tokenize(i)
    filtered_tweets = [stemmer.stem(word) for word in words if word is not stop_words and word.isalpha()]
    nested_list.append(filtered_tweets)
word_columns_df = pd.DataFrame.from_records(nested_list)
word_columns_df.head()
(X_train, X_test, y_train, y_test) = train_test_split(word_columns_df, train.target, test_size=0.2, random_state=1)
flatten = [item for sublist in nested_list for item in sublist]
unique_words = pd.Series(flatten).value_counts()
frequent_words = unique_words[0:word_size]
tweet_ids = list(range(0, word_size))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=tweet_ids)
vocab.index.name = 'WORD_ID'
word_index = pd.Index(vocab.VOCAB_WORD)

def create_sparse_matrix(df, indexed_words, labels):
    rows = df.shape[0]
    cols = df.shape[1]
    word_set = set(indexed_words)
    my_dict = []
    for i in range(rows):
        for j in range(cols):
            word = df.iat[i, j]
            if word in word_set:
                ID = df.index[i]
                tweet_id = indexed_words.get_loc(word)
                category = labels.at[ID]
                item = {'LABEL': category, 'TWEET_ID': ID, 'FREQ': 1, 'WORD_ID': tweet_id}
                my_dict.append(item)
    return pd.DataFrame(my_dict)
sparse_train = create_sparse_matrix(X_train, word_index, y_train)
train_grouped = sparse_train.groupby(['TWEET_ID', 'WORD_ID', 'LABEL']).sum().reset_index()
train_grouped.head()
sparse_test = create_sparse_matrix(X_test, word_index, y_test)
test_grouped = sparse_test.groupby(['TWEET_ID', 'WORD_ID', 'LABEL']).sum().reset_index()
test_grouped.head()
train_grouped.LABEL.value_counts()

def make_full_matrix(matrix):
    matrix = np.array(matrix)
    col = ['TWEET_ID'] + ['TARGET'] + list(range(0, word_size))
    index_names = np.unique(matrix[:, 0])
    complete_matrix = pd.DataFrame(index=index_names, columns=col)
    complete_matrix.fillna(0, inplace=True)
    for i in range(matrix.shape[0]):
        doc_num = matrix[i][0]
        tweet_id = matrix[i][1]
        label = matrix[i][2]
        frequency = matrix[i][3]
        complete_matrix.at[doc_num, 'TWEET_ID'] = doc_num
        complete_matrix.at[doc_num, 'TARGET'] = label
        complete_matrix.at[doc_num, tweet_id] = frequency
    complete_matrix.set_index('TWEET_ID', inplace=True)
    return complete_matrix
train_data = make_full_matrix(train_grouped)
train_data.head()
test_data = make_full_matrix(test_grouped)
test_data.head()
prob_disaster = train_data.TARGET.sum() / train_data.TARGET.size
print(f'Probability of a tweet indicating a disaster : {prob_disaster}')
print(f'Probability of a tweet indicating not disaster : {1 - prob_disaster}')
all_train_features = train_data.loc[:, train_data.columns != 'TARGET']
total_tweets_sum = all_train_features.sum(axis=1)
total_tweets = total_tweets_sum.sum()
print(f'Total tweets : {total_tweets}')
disaster_only_sum = total_tweets_sum[train_data['TARGET'] == 1]
total_disaster_tweets = disaster_only_sum.sum()
print(f'Total disaster tweets : {total_disaster_tweets}')
not_disaster_only_sum = total_tweets_sum[train_data['TARGET'] == 0]
total_not_disaster_tweets = not_disaster_only_sum.sum()
print(f'Total not disaster tweets : {total_not_disaster_tweets}')
tweets = [total_disaster_tweets, total_not_disaster_tweets]
fig = px.pie(values=tweets, names=['Disaster tweets', 'Not Disaster tweets'], title='Pie Chart : Disaster & Not Disaster')
fig.show()
disaster_only_sum = all_train_features.loc[train_data.TARGET == 1]
disaster_tok_sum = disaster_only_sum.sum(axis=0) + 1
print(f'Total Disaster Tokens : {disaster_tok_sum.sum()}')
not_disaster_only_sum = all_train_features.loc[train_data.TARGET == 0]
non_disaster_tok_sum = not_disaster_only_sum.sum(axis=0) + 1
print(f'Total Non Disaster Tokens : {non_disaster_tok_sum.sum()}')
prob_tok_disaster = disaster_tok_sum / (total_disaster_tweets + word_size)
prob_tok_not_disaster = non_disaster_tok_sum / (total_not_disaster_tweets + word_size)
prob_tok_all = all_train_features.sum(axis=0) / total_tweets
X_test = test_data.loc[:, test_data.columns != 'TARGET']
y_test = test_data['TARGET']
disaster = X_test.dot(np.log(prob_tok_disaster)) + np.log(prob_disaster)
non_disaster = X_test.dot(np.log(prob_tok_not_disaster)) + np.log(1 - prob_disaster)
predict = disaster > non_disaster
print(f'Tweets correctly predicted {(y_test == predict).sum()}')
print(f'Tweets incorrectly predicted {X_test.shape[0] - (y_test == predict).sum()}')
print(f'\nOur accuracy : {(y_test == predict).sum() / len(X_test) * 100:.3}%')
chart = pd.DataFrame(disaster)
chart.columns = ['Disaster']
chart['Not Disaster'] = non_disaster
chart['Target'] = predict
import plotly.graph_objects as go
linedata = np.linspace(start=-200, stop=1, num=200)
fig = go.Figure()
fig = px.scatter(chart, y='Disaster', x='Not Disaster', color=predict, opacity=0.5)
fig.add_trace(go.Scatter(x=linedata, y=linedata, mode='lines', name='Boundary'))
fig.update_traces(marker_size=5.5, line_width=4, line_color='midnightblue')
fig.update_layout(title='Correctly Predicted vs Incorrectly Predicted')
fig.show()