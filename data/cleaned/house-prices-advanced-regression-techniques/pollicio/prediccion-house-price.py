import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = (14, 8)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_tests():
    unittest.main(argv=[''], verbosity=1, exit=False)
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_train
df_train
df = df_train['SalePrice'].describe()
df
sns.distplot(df_train['SalePrice'])
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), s=32)
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
(f, ax) = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
corrmat = df_train.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
corrmat = df_train.corr()
(f, ax) = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.pairplot(df_train[cols], size=4)
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
x = df_train['GrLivArea']
y = df_train['SalePrice']
x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = (14, 8)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_tests():
    unittest.main(argv=[''], verbosity=1, exit=False)

def loss(h, y):
    sq_error = (h - y) ** 2
    n = len(y)
    return 1.0 / (2 * n) * sq_error.sum()

class TestLoss(unittest.TestCase):

    def test_zero_h_zero_y(self):
        self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([0])), 0)

    def test_one_h_zero_y(self):
        self.assertAlmostEqual(loss(h=np.array([1]), y=np.array([0])), 0.5)

    def test_two_h_zero_y(self):
        self.assertAlmostEqual(loss(h=np.array([2]), y=np.array([0])), 2)

    def test_zero_h_one_y(self):
        self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([1])), 0.5)

    def test_zero_h_two_y(self):
        self.assertAlmostEqual(loss(h=np.array([0]), y=np.array([2])), 2)
run_tests()

class LinearRegression:

    def predict(self, X):
        return np.dot(X, self._W)

    def _gradient_descent_step(self, X, targets, lr):
        predictions = self.predict(X)
        error = predictions - targets
        gradient = np.dot(X.T, error) / len(X)
        self._W -= lr * gradient

    def fit(self, X, y, n_iter=100000, lr=0.01):
        self._W = np.zeros(X.shape[1])
        self._cost_history = []
        self._w_history = [self._W]
        for i in range(n_iter):
            prediction = self.predict(X)
            cost = loss(prediction, y)
            self._cost_history.append(cost)
            self._gradient_descent_step(x, y, lr)
            self._w_history.append(self._W.copy())
        return self

class TestLinearRegression(unittest.TestCase):

    def test_find_coefficients(self):
        clf = LinearRegression()