import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

train_data = pd.read_csv('data/input/digit-recognizer/train.csv')
test_data = pd.read_csv('data/input/digit-recognizer/test.csv')
train_data.shape
test_data.shape
train_data.head()
test_data.head()
train_data.isnull().sum().head(10)
test_data.isnull().sum().head(10)
test_data.describe()
train_data.describe()
print('Dimensions: ', test_data.shape, '\n')
print(test_data.info())
test_data.head()
print('Dimensions: ', train_data.shape, '\n')
print(train_data.info())
train_data.head()
print(train_data.columns)
print(test_data.columns)
order = list(np.sort(train_data['label'].unique()))
print(order)
sns.countplot(train_data['label'])
plt.plot(figure=(16, 10))
g = sns.countplot(train_data['label'], palette='icefire')
plt.title('NUmber of digit classes')
train_data.label.astype('category').value_counts()
four = train_data.iloc[3, 1:]
four.shape
four = four.values.reshape(28, 28)
plt.imshow(four, cmap='gray')
plt.title('Digit 4')
seven = train_data.iloc[6, 1:]
seven.shape
seven = seven.values.reshape(28, 28)
plt.imshow(seven, cmap='gray')
plt.title('Digit 7')
round(train_data.drop('label', axis=1).mean(), 2)
y = train_data['label']
X = train_data.drop(columns='label')
print(train_data.shape)
X = X / 255.0
test_data = test_data / 255.0
print('X:', X.shape)
print('test_data:', test_data.shape)
from sklearn.preprocessing import scale
X_scaled = scale(X)
(X_train, X_test, y_train, y_test) = train_test_split(X_scaled, y, test_size=0.3, train_size=0.2, random_state=10)

class Dataset:

    def __init__(self):
        self.train_data = pd.read_csv('data/input/digit-recognizer/train.csv')
        self.test_data = pd.read_csv('data/input/digit-recognizer/test.csv')

    def preprocess_data(self, image_data, labels):
        image_data = np.array(image_data) / 255.0
        labels = np.array(labels) if labels is not None else None
        return (image_data, labels)

    def classify_wise_data(self, image_data, labels):
        data = {}
        for i in range(len(np.unique(labels))):
            data[i] = []
        for i in range(image_data.shape[0]):
            data[labels[i]].append(image_data[i])
        for k in data.keys():
            data[k] = np.array(data[k])
        return data

    def load_data(self, type='train'):
        if type == 'train':
            (image_data, labels) = (self.train_data.drop(columns='label'), self.train_data['label'])
        else:
            (image_data, labels) = (self.test_data, None)
        (image_data, labels) = self.preprocess_data(image_data, labels)
        if labels is None:
            return image_data
        m = image_data.shape[0]
        image_data = image_data.reshape(m, -1)
        data = self.classify_wise_data(image_data, labels)
        return (image_data, labels, data)
dataset = Dataset()
(train_image_data, train_labels, train_data) = dataset.load_data('train')
val_image_data = dataset.load_data('val')
(train_image_data.shape, train_labels.shape)
(train_data[0].shape, train_data[1].shape)

class SVM:

    def __init__(self, total_class, C=1.0):
        self.C = C
        self.W = 0
        self.b = 0
        self.total_class = total_class

    def hinge_loss(self, W, b, X, Y):
        loss = 0.0
        loss += 0.5 * np.dot(W, W.T)
        m = X.shape[0]
        for i in range(m):
            ti = Y[i] * (np.dot(W, X[i].T) + b)
            loss += self.C * max(0, 1 - ti)
        return loss[0][0]

    def fit(self, X, Y, batch_size=50, learning_rate=0.001, max_iter=500):
        print(X.shape, Y.shape)
        num_features = X.shape[1]
        num_samples = X.shape[0]
        n = learning_rate
        c = self.C
        W = np.zeros((1, num_features))
        bias = 0
        losses = []
        for i in tqdm(range(max_iter)):
            l = self.hinge_loss(W, bias, X, Y)
            losses.append(l)
            ids = np.arange(num_samples)
            np.random.shuffle(ids)
            for batch_start in range(0, num_samples, batch_size):
                gradw = 0
                gradb = 0
                for j in range(batch_start, batch_start + batch_size):
                    if j < num_samples:
                        i = ids[j]
                        ti = Y[i] * (np.dot(W, X[i].T) + bias)
                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c * Y[i] * X[i]
                            gradb += c * Y[i]
                W = W - n * W + n * gradw
                bias = bias + n * gradb
        self.W = W
        self.b = bias
        return (W, bias, losses)

    def get_data_pair(self, data1, data2):
        (len1, len2) = (data1.shape[0], data2.shape[0])
        samples = len1 + len2
        features = data1.shape[1]
        data_pair = np.zeros((samples, features))
        data_labels = np.zeros((samples,))
        data_pair[:len1, :] = data1
        data_pair[len1:, :] = data2
        data_labels[:len1] = -1
        data_labels[len1:] = +1
        return (data_pair, data_labels)

    def train(self, data, batch_size=50, learning_rate=1e-05, max_iter=500):
        self.svm_classifiers = {}
        for i in range(self.total_class):
            self.svm_classifiers[i] = {}
            for j in range(i + 1, self.total_class):
                (xpair, ypair) = self.get_data_pair(data[i], data[j])