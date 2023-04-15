import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
df_train = pd.read_csv('data/input/nlp-getting-started/train.csv')
df_train

print()

df_train['char_count'] = df_train['text'].apply(lambda x: len(x))
df_train['list_word'] = df_train['text'].str.split()
df_train['word_count'] = df_train['list_word'].apply(lambda x: len(x))
df_train['unique_word_count'] = df_train['list_word'].apply(lambda x: len(set(x)))
df_train['mean_word_length'] = df_train['list_word'].apply(lambda x: [len(i) for i in x]).apply(lambda x: np.mean(x))

def find_specific_words(pattern, string):
    words = re.findall(pattern, string)
    return [x[0] for x in words]
stop = set(stopwords.words('english'))
df_train['stopword_count'] = df_train['list_word'].apply(lambda x: len([word for word in x if word in stop]))
df_train['unstopword_count'] = df_train['list_word'].apply(lambda x: len([word for word in x if word not in stop]))
punctuations = string.punctuation
df_train['punc_count'] = df_train['text'].apply(lambda x: len([c for c in x if c in punctuations]))
url_pattern = '(?i)\\b((?:https?://|www\\d{0,3}[.]|[a-z0-9.\\-]+[.][a-z]{2,4}/)(?:[^\\s()<>]+|\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\))+(?:\\(([^\\s()<>]+|(\\([^\\s()<>]+\\)))*\\)|[^\\s`!()\\[\\]{};:\'\\".,<>?«»“”‘’]))'
df_train['url_count'] = df_train['text'].apply(lambda x: len(find_specific_words(url_pattern, x)))
hashtag_pattern = '#([A-Za-z0-9]+)'
df_train['hashtag_count'] = df_train['text'].apply(lambda x: len(find_specific_words(hashtag_pattern, x)))
mention_pattern = '@([A-Za-z0-9]+)'
df_train['mention_count'] = df_train['text'].apply(lambda x: len(find_specific_words(mention_pattern, x)))
df_train['upper_case_count'] = df_train['text'].apply(lambda x: len([c for c in x if c.isupper()]))
df_train['lower_case_count'] = df_train['text'].apply(lambda x: len([c for c in x if c.islower()]))


print(df_train.shape)

def draw_diverge_colors_histogram(x_label, bins, title, log=False):
    (fig, ax) = plt.subplots(figsize=(15, 5.5))
    df_count = df_train[[x_label, 'target']]
    target = pd.get_dummies(df_count['target'], prefix='target')
    df_count = pd.merge(df_count, target, left_index=True, right_index=True)
    bin_space = np.linspace(df_count[x_label].min(), df_count[x_label].max(), bins + 1)
    df_count['bins'] = np.digitize(df_count[x_label], bin_space) - 1
    df_count['bins'] = df_count['bins'].replace(to_replace=bins, value=bins - 1)
    color_bins = df_count.groupby('bins')[['target_0', 'target_1']].apply(sum)
    set_get_bins = set(df_count['bins'])
    set_actual_bins = set(range(bins))
    set_diff = set_actual_bins.difference(set_get_bins)
    if len(set_diff) != 0:
        for idx in set_diff:
            color_bins.loc[idx, :] = 0
        color_bins = color_bins.sort_index()
    color_bins['total'] = color_bins['target_0'] + color_bins['target_1']
    text_pos = color_bins['total'].max()
    color_bins['percentage'] = round(color_bins['target_1'] / color_bins['total'] * 100 - color_bins['target_0'] / color_bins['total'] * 100, 1)
    ax = sns.histplot(df_count[x_label], bins=bins, edgecolor='white')
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    for i in range(bins):
        if color_bins['total'][i] != 0:
            if log:
                fix_pos = np.log(ax.patches[i].get_height()) / np.log(text_pos) + 0.08
            else:
                fix_pos = ax.patches[i].get_height() / text_pos + 0.03
            if color_bins['percentage'][i] <= 0.0:
                ax.text((bin_space[i] + bin_space[i + 1]) / 2, fix_pos, str(color_bins['percentage'][i]) + '%', color='black', horizontalalignment='center', verticalalignment='center', transform=trans)
            else:
                ax.text((bin_space[i] + bin_space[i + 1]) / 2, fix_pos, str(color_bins['percentage'][i]) + '%', fontweight='bold', color='black', horizontalalignment='center', verticalalignment='center', transform=trans)
        if color_bins['percentage'][i] < -60.0:
            color = cmap(0.1)
        elif (color_bins['percentage'][i] >= -60.0) & (color_bins['percentage'][i] < -20.0):
            color = cmap(0.2)
        elif (color_bins['percentage'][i] >= -20.0) & (color_bins['percentage'][i] < -10.0):
            color = cmap(0.3)
        elif (color_bins['percentage'][i] >= -10.0) & (color_bins['percentage'][i] < 0.0):
            color = cmap(0.4)
        elif (color_bins['percentage'][i] >= 0.0) & (color_bins['percentage'][i] < 10.0):
            color = cmap(0.6)
        elif (color_bins['percentage'][i] >= 10.0) & (color_bins['percentage'][i] < 20.0):
            color = cmap(0.7)
        elif (color_bins['percentage'][i] >= 20.0) & (color_bins['percentage'][i] < 60.0):
            color = cmap(0.8)
        else:
            color = cmap(0.9)
        ax.patches[i].set_facecolor(color)
    if log:
        ax.set_yscale('log')
        ax.set_ylabel('Log Count')
    ax.set_ylim(bottom=1)
    ax.set_xticks(bin_space)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(title + ' Based on Histogram', fontsize=15, y=1.1)


def draw_diverge_colors_ver_bar(x_label, title, log=False, text_range=None, xticks=None):
    (fig, ax) = plt.subplots(figsize=(15, 5.5))
    df_count = df_train[[x_label, 'target']]
    target = pd.get_dummies(df_count['target'], prefix='target')
    df_count = pd.merge(df_count, target, left_index=True, right_index=True)
    color_bars = df_count.groupby(x_label)[['target_0', 'target_1']].apply(sum)
    set_actual = set(color_bars.index.tolist())
    set_expected = set(np.arange(color_bars.index[0], color_bars.index[-1] + 1))
    empty_bars = set_expected.difference(set_actual)
    if empty_bars:
        for idx in empty_bars:
            color_bars.loc[idx, :] = 0
        color_bars = color_bars.sort_index()
    color_bars['total'] = color_bars['target_0'] + color_bars['target_1']
    text_pos = color_bars['total'].max()
    color_bars['percentage'] = round(color_bars['target_1'] / color_bars['total'] * 100 - color_bars['target_0'] / color_bars['total'] * 100, 1)
    color_bars = color_bars.reset_index()
    ax = sns.barplot(data=color_bars, x=x_label, y='total')
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if text_range:
        text_range = set(np.arange(text_range[0], text_range[1] + 1))
    else:
        text_range = set(np.arange(0, color_bars.shape[0]))
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    for i in range(color_bars.shape[0]):
        if color_bars['total'][i] != 0 and color_bars[x_label][i] in text_range:
            if log:
                fix_pos = np.log(ax.patches[i].get_height()) / np.log(text_pos) + 0.08
            else:
                fix_pos = ax.patches[i].get_height() / text_pos + 0.03
            if color_bars['percentage'][i] <= 0.0:
                ax.text(i, fix_pos, str(color_bars['percentage'][i]) + '%', color='black', horizontalalignment='center', verticalalignment='center', transform=trans)
            else:
                ax.text(i, fix_pos, str(color_bars['percentage'][i]) + '%', fontweight='bold', color='black', horizontalalignment='center', verticalalignment='center', transform=trans)
        if color_bars['percentage'][i] < -60.0:
            color = cmap(0.1)
        elif (color_bars['percentage'][i] >= -60.0) & (color_bars['percentage'][i] < -20.0):
            color = cmap(0.2)
        elif (color_bars['percentage'][i] >= -20.0) & (color_bars['percentage'][i] < -10.0):
            color = cmap(0.3)
        elif (color_bars['percentage'][i] >= -10.0) & (color_bars['percentage'][i] < 0.0):
            color = cmap(0.4)
        elif (color_bars['percentage'][i] >= 0.0) & (color_bars['percentage'][i] < 10.0):
            color = cmap(0.6)
        elif (color_bars['percentage'][i] >= 10.0) & (color_bars['percentage'][i] < 20.0):
            color = cmap(0.7)
        elif (color_bars['percentage'][i] >= 20.0) & (color_bars['percentage'][i] < 60.0):
            color = cmap(0.8)
        else:
            color = cmap(0.9)
        ax.patches[i].set_facecolor(color)
    if log:
        ax.set_yscale('log')
        ax.set_ylabel('Log Count')
    ax.set_ylim(bottom=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xticks:
        for (ind, label) in enumerate(ax.get_xticklabels()):
            if ind in xticks:
                label.set_visible(True)
            else:
                label.set_visible(False)
    plt.title(title + ' Based on Bar Chart', fontsize=15, y=1.1)


def draw_two_hor_bar(target, suptitle):
    (fig, axs) = plt.subplots(figsize=(17, 8))
    for x in range(2):
        p = 121 + x
        ax = plt.subplot(p)
        df_target = pd.DataFrame(target[x]).reset_index()
        df_target.columns = ['Word', 'Count']
        total = df_target['Count'].max()
        ax = sns.barplot(data=df_target, x='Count', y='Word', orient='h')
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        cmap = sns.color_palette('coolwarm', as_cmap=True)
        for i in range(10):
            if i == 0:
                ax.text(df_target['Count'][i] / total + 0.03, i, str(df_target['Count'][i]), fontweight='bold', color='black', horizontalalignment='center', verticalalignment='center', transform=trans)
                ax.patches[i].set_facecolor('#937DC2')
            else:
                ax.text(df_target['Count'][i] / total + 0.03, i, str(df_target['Count'][i]), color='black', horizontalalignment='left', verticalalignment='center', transform=trans)
                ax.patches[i].set_facecolor('#FF8AA5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', which='major', pad=5)
        ax.tick_params(left=False, bottom=False)
        ax.set(xticklabels=[])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.title('Target {}'.format(x), y=1.02, loc='center', fontsize=15)
    plt.suptitle('Top 10 ' + suptitle, fontsize=20, y=1.04)
    plt.tight_layout()

draw_diverge_colors_histogram('char_count', 15, 'Character Count')
draw_diverge_colors_histogram('word_count', 15, 'Word Count')
draw_diverge_colors_ver_bar('word_count', 'Word Count', text_range=[2, 28])
draw_diverge_colors_histogram('unique_word_count', 14, 'Unique Word Count')
draw_diverge_colors_ver_bar('unique_word_count', 'Unique Word Count', text_range=[2, 26])
draw_diverge_colors_histogram('mean_word_length', 12, 'Mean Word Length', log=True)
draw_diverge_colors_ver_bar('stopword_count', 'Stopword Count')
draw_diverge_colors_ver_bar('unstopword_count', 'Unstopword Count', text_range=[2, 20])
draw_diverge_colors_ver_bar('punc_count', 'Punctuation Count', log=True, text_range=[10000, 10000])
draw_diverge_colors_ver_bar('url_count', 'URL Count', log=True)
draw_diverge_colors_ver_bar('hashtag_count', 'Hashtag Count', log=True)
draw_diverge_colors_ver_bar('mention_count', 'Mention Count', log=True)
draw_diverge_colors_ver_bar('upper_case_count', 'Upper Case Count', log=True, text_range=[10000, 10000], xticks=[0, 7, 18, 31, 74, 118])
draw_diverge_colors_ver_bar('lower_case_count', 'Lower Case Count', text_range=[10000, 10000], xticks=[0, 95, 99, 119])

def get_top_count_vectorizer(sentence_list, ngram, n):
    vectorizer = CountVectorizer(ngram_range=(ngram, ngram), stop_words='english')
    bag_of_words = vectorizer.fit_transform(sentence_list)
    sort_vocab = sorted(vectorizer.vocabulary_.items())
    list_vocab = [word[0] for word in sort_vocab]
    df_vectorizer = pd.DataFrame(bag_of_words.todense(), columns=list_vocab)
    df_vectorizer['target'] = df_train['target']
    df_vectorizer_0 = df_vectorizer[df_vectorizer['target'] == 0].drop('target', axis=1)
    target_0_top_count = df_vectorizer_0.sum().sort_values(ascending=False)
    df_vectorizer_1 = df_vectorizer[df_vectorizer['target'] == 1].drop('target', axis=1)
    target_1_top_count = df_vectorizer_1.sum().sort_values(ascending=False)
    return (target_0_top_count[:n], target_1_top_count[:n])
(target_0_top_count, target_1_top_count) = get_top_count_vectorizer(df_train['text'], ngram=1, n=10)
draw_two_hor_bar([target_0_top_count, target_1_top_count], 'Unigram')
(target_0_top_count, target_1_top_count) = get_top_count_vectorizer(df_train['text'], ngram=2, n=10)
draw_two_hor_bar([target_0_top_count, target_1_top_count], 'Bigram')
(target_0_top_count, target_1_top_count) = get_top_count_vectorizer(df_train['text'], ngram=3, n=10)
draw_two_hor_bar([target_0_top_count, target_1_top_count], 'Trigram')