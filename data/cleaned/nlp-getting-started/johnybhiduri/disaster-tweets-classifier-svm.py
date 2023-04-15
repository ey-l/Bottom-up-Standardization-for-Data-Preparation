import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/input/nlp-getting-started/train.csv')
test = pd.read_csv('data/input/nlp-getting-started/test.csv')
X = train['text']
Y = train['target']
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=1)
test_split = test['text']
cv = CountVectorizer()
features = cv.fit_transform(x_train)
class_weight = {0: 1, 1: 2}
model = svm.SVC(kernel='rbf', class_weight=class_weight)