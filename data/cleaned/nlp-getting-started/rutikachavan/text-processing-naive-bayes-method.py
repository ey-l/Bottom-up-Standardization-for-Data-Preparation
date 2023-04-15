from re import sub
import pandas as pd
A = pd.read_csv('data/input/nlp-getting-started/train.csv')
A
A.isna().sum()
A = A.drop(labels=['keyword', 'location'], axis=1)
A
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

def prep_txt(w):
    import re
    q = re.sub('[^a-zA-Z0-9 ]', '', w)
    q = q.lower()
    q = q.split(' ')
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    r = ''
    for i in q:
        if i not in sw:
            t = wnl.lemmatize(i)
            t = ps.stem(t)
            r = r + ' ' + t
    return r
Q = []
for i in A.text:
    Q.append(prep_txt(i))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(Q).toarray()
Y = A.target
X
from sklearn.model_selection import train_test_split
(xtrain, xtest, ytrain, ytest) = train_test_split(X, Y, test_size=0.2, random_state=31)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()