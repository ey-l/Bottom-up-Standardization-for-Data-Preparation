import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
_input2 = pd.read_csv('data/input/nlp-getting-started/sample_submission.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input1.head()
_input1.info()
_input1['keyword'].unique()
_input1['location'].unique()
data = [_input1.groupby(['target']).count()['id'][0], _input1.groupby(['target']).count()['id'][1]]
colors = sns.color_palette('pastel')
labels = ['not-disaster', 'disaster']
plt.pie(data, colors=colors, labels=labels, autopct='%0.0f%%')
palette = sns.color_palette('magma')
data = pd.DataFrame(_input1[_input1['target'] == 1].groupby(['keyword']).count().id.sort_values(ascending=False)[:20]).reset_index()
fig = plt.figure(figsize=(30, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(data.keyword, data.id)
ax.set_ylabel('Count')
ax.set_xlabel('A particular keywords from the disaster tweet (may be blank)')
data = pd.DataFrame(_input1[_input1['target'] == 0].groupby(['keyword']).count().id.sort_values(ascending=False)[:20]).reset_index()
fig = plt.figure(figsize=(30, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(data.keyword, data.id)
ax.set_ylabel('Count')
ax.set_xlabel('A particular keywords from the not-disaster tweet (may be blank)')
data = pd.DataFrame(_input1.groupby('location').count().id.sort_values(ascending=False)[:20]).reset_index()
fig = plt.figure(figsize=(30, 6))
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(data.location, data.id)
ax.set_ylabel('Count')
ax.set_xlabel('The location the tweet was sent from (may also be blank)')
for (label, cmap) in zip([1, 0], ['magma', 'winter']):
    text = _input1.query('target == @label')['text'].str.cat(sep=' ')
    plt.figure(figsize=(10, 6))
    wc = WordCloud(width=1000, height=600, background_color='#f8f8f8', colormap=cmap)
    wc.generate_from_text(text)
    plt.imshow(wc)
    plt.axis('off')
    plt.title(f'Words Commonly Used in which target is ${label}$', size=20)
_input1 = _input1.replace(regex={'%20': ' '}, inplace=False)
_input1 = _input1.replace(regex={'AFAIK': 'As Far As I Know', ' AFK ': ' Away From Keyboard ', ' ASAP ': ' As Soon As Possible ', ' ATK ': ' At The Keyboard ', ' ATM ': ' At The Moment ', ' A3 ': ' Anytime, Anywhere, Anyplace ', ' BAK ': ' Back At Keyboard ', ' BBL ': ' Be Back Later ', ' BBS ': ' Be Back Soon ', ' BFN ': ' Bye For Now ', ' B4N ': ' Bye For Now ', ' BRB ': ' Be Right Back ', ' BRT ': ' Be Right There ', ' BTW ': ' By The Way ', ' B4 ': ' Before ', ' B4N ': ' Bye For Now ', ' CU ': ' See You ', ' CUL8R ': ' See You Later ', ' CYA ': ' See You ', ' FAQ ': ' Frequently Asked Questions ', ' FC ': ' Fingers Crossed ', ' FWIW ': " For What It's Worth ", ' FYI ': ' For Your Information ', ' GAL ': ' Get A Life ', ' GG ': ' Good Game ', ' GN ': ' Good Night ', ' GMTA ': ' Great Minds Think Alike ', ' GR8 ': ' Great! ', ' G9 ': ' Genius ', ' IC ': ' I See ', ' ICQ ': ' I Seek you ', ' ILU ': ' I Love You ', ' IMHO ': ' In My Honest ', ' IMO ': ' In My Opinion ', ' IOW ': ' In Other Words ', ' IRL ': ' In Real Life ', ' KISS ': ' Keep It Simple, Stupid ', ' LDR ': ' Long Distance Relationship ', ' LMAO ': ' Laugh My Ass ', ' LOL ': ' Laughing Out Loud ', ' LTNS ': ' Long Time No See ', ' L8R ': ' Later ', ' MTE ': ' My Thoughts Exactly ', ' M8 ': ' Mate ', ' NRN ': ' No Reply Necessary ', ' OIC ': ' Oh I See ', ' PITA ': ' Pain In The Ass ', ' PRT ': ' Party ', ' PRW ': ' Parents Are Watching ', ' ROFL ': ' Rolling On The Floor Laughing ', ' ROFLOL ': ' Rolling On The Floor Laughing Out Loud ', ' ROTFLMAO ': ' Rolling On The Floor Laughing My Ass ', ' SK8 ': ' Skate ', ' STATS ': ' Your sex and age ', ' ASL ': ' Age, Sex, Location ', ' THX ': ' Thank You ', ' TTFN ': ' Ta-Ta For Now! ', ' TTYL ': ' Talk To You Later ', ' U ': ' You ', ' U2 ': ' You Too', ' U4E ': ' Yours For Ever ', ' WB ': ' Welcome Back ', ' WTF ': ' What The Fuck ', ' WTG ': ' Way To Go! ', ' WUF ': ' Where Are You From? ', ' W8 ': ' Wait... '}, inplace=False)
_input1 = _input1.replace(regex={'https?://\\S+': ' ', '<.*?>': ' ', '\\d+': ' ', '#\\w+': ' ', '[^a-zA-Z]': ' ', 'http\\S+': ' '}, inplace=False)
sw = stopwords.words('english')
_input1['text'][2]
_input1['text'] = _input1['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in sw]))
_input1['text'][2]
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(_input1['text'])
target = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
y = _input1['target']
(x_train, x_val, y_train, y_val) = train_test_split(target, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
print(x_train.shape)
print(x_val.shape)
mlp = MLPClassifier(random_state=0, early_stopping=True, verbose=2)