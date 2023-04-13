import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/nlp-getting-started/train.csv')
_input0 = pd.read_csv('data/input/nlp-getting-started/test.csv')
X = _input1['text']
Y = _input1['target']
(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=1)
test_split = _input0['text']
cv = CountVectorizer()
features = cv.fit_transform(x_train)
class_weight = {0: 1, 1: 2}
model = svm.SVC(kernel='rbf', class_weight=class_weight)