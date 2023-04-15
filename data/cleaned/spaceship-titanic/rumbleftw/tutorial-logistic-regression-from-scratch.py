import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class LogisticRegression:

    def __init__(self, learningRate=0.0001, iterations=1000):
        """
        Constructor, the default lr and n_iters are declared
        """
        self.m = None
        self.n = None
        self.weights = None
        self.bias = None
        self.iterations = iterations
        self.learningRate = learningRate

    def lossFunction(self, y, yHat):
        """
        Binary cross entropy as loss function
        """
        yZeroLoss = y * np.log(yHat + 1e-09)
        yOneLoss = (1 - y) * np.log(1 - yHat + 1e-09)
        return -np.mean(yZeroLoss + yOneLoss)

    def sigmoid(self, x):
        """
        The sigmoid function for calculating the probability
        """
        return 1.0 / (1 + np.exp(-x))

    def gradientDescent(self, X, y):
        """
        Gradient descent, which will update the weights for self.iterations until a minima is reached
        """
        for iteration in range(self.iterations):
            y = y.astype(np.float64)
            yHat = self.predict(X)
            dw = np.matmul(X.transpose(), yHat - y)
            dw = np.array([np.mean(gradient) for gradient in dw])
            db = np.mean(yHat - y)
            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db
            print(f'Iteration: {iteration + 1}, Loss: {self.lossFunction(yHat, y)}')

    def fit(self, X_train, y_train):
        """
        Training of model on data X_train and y_train
        """
        (self.m, self.n) = X_train.shape
        self.weights = np.zeros(self.n, dtype=np.float64)
        self.bias = 0
        self.gradientDescent(X_train, y_train)
        self.evaluate(X_train, y_train)

    def predict(self, X):
        """
        Predicting the probability of labels by a given features. If probability >= threshold the label is 1. Else 0.
        """
        return np.array([self.sigmoid(np.dot(x, self.weights) + self.bias) for x in X])

    def predictLabels(self, X, threshold=0.5):
        """
        Predicting the labels by a given features
        """
        preds = self.predict(X)
        return np.array([1 if value >= threshold else 0 for value in preds])

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Calculating the accuracy and precision on a given data.
        """
        preds = self.predictLabels(X_test)
        N = y_test.shape[-1]
        accuracy = (y_test == preds).sum() / N
        TP = ((preds == 1) & (y_test == 1)).sum()
        FP = ((preds == 1) & (y_test == 0)).sum()
        precision = TP / (TP + FP)
        print(f'Accuracy: {accuracy}, Precision: {precision}')
dataTrain = pd.read_csv('data/input/spaceship-titanic/train.csv')
dataTest = pd.read_csv('data/input/spaceship-titanic/test.csv')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
sns.heatmap(ax=axes[0], yticklabels=False, data=dataTrain.isnull(), cbar=False, cmap='rainbow_r')
sns.heatmap(ax=axes[1], yticklabels=False, data=dataTest.isnull(), cbar=False, cmap='tab20c')
axes[0].set_title('Heatmap of missing values in training data')
axes[1].set_title('Heatmap of missing values in testing data')
plt.suptitle('Null values in datasets')

plt.figure(figsize=(18, 9))
sns.heatmap(dataTrain.corr(), cmap='YlGnBu', annot=True)
plt.title('Correlation between features')

plt.figure(figsize=(12, 8))
colors = sns.color_palette('pastel')
plt.pie([item / len(dataTrain.HomePlanet) for item in dataTrain.HomePlanet.value_counts()], labels=['Earth', 'Europa', 'Mars'], colors=colors, autopct='%.0f%%')
plt.title('Distribution of Individuals based on HomePlanet')

trainAge = dataTrain.copy()
testAge = dataTest.copy()
trainAge['type'] = 'Train'
testAge['type'] = 'Test'
ageDf = pd.concat([trainAge, testAge])
fig = px.histogram(data_frame=ageDf, x='Age', color='type', color_discrete_sequence=['#FFA500', '#87CEEB'], marginal='box', nbins=100, template='plotly_white')
fig.update_layout(title='Distribution of Age', title_x=0.5)
fig.show()
idCol = dataTest.PassengerId.to_numpy()
dataTrain.set_index('PassengerId', inplace=True)
dataTest.set_index('PassengerId', inplace=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataTrain = pd.DataFrame(imputer.fit_transform(dataTrain), columns=dataTrain.columns, index=dataTrain.index)
dataTest = pd.DataFrame(imputer.fit_transform(dataTest), columns=dataTest.columns, index=dataTest.index)
dataTrain = dataTrain.reset_index(drop=True)
dataTest = dataTest.reset_index(drop=True)
dataTrain.Transported = dataTrain.Transported.astype('int')
dataTrain.VIP = dataTrain.VIP.astype('int')
dataTrain.CryoSleep = dataTrain.CryoSleep.astype('int')
dataTrain.drop(columns=['Cabin', 'Name'], inplace=True)
dataTest.drop(columns=['Cabin', 'Name'], inplace=True)
dataTrain = pd.get_dummies(dataTrain, columns=['HomePlanet', 'CryoSleep', 'Destination'])
dataTest = pd.get_dummies(dataTest, columns=['HomePlanet', 'CryoSleep', 'Destination'])
dataTrain.head()
y_train = dataTrain.pop('Transported').to_numpy()
X_train = dataTrain.to_numpy()
X_test = dataTest.to_numpy()
(X_train.shape, y_train.shape, X_test.shape)
lrClassifier = LogisticRegression(learningRate=0.001, iterations=17)