import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

class NaiveBayes:

    def __init__(self, path, ratio, seed):
        self.path = path
        self.__RATIO = ratio
        self.__SEED = seed

    def import_data(self):
        ds = pd.read_csv(self.path, skiprows=1, header=None)
        self.x = ds.iloc[:, :-1]
        self.y = ds.iloc[:, -1]
        (self.x_train, self.x_test, self.y_train, self.y_test) = train_test_split(self.x, self.y, random_state=SEED, train_size=RATIO)
        self.y_hat = np.zeros(self.y_test.shape[0], dtype='int')

    @staticmethod
    def log_gaussian(x, mu, variance):
        return np.lib.scimath.log(scipy.stats.norm(mu, variance).pdf(x))

    @staticmethod
    def train_model(x, y):
        indices = [np.repeat(np.unique(y), len(np.unique(y))), np.array(['Mean', 'STD'] * len(np.unique(y)))]
        df = pd.DataFrame(np.empty((len(np.unique(y)) * 2, x.shape[1])), index=indices, columns=list(range(x.shape[1])))
        for val in np.unique(y):
            means = x[y == val].mean()
            std = x[y == val].std()
            df.loc[(val, 'Mean'), :] = means
            df.loc[(val, 'STD'), :] = std
        print('Data Summary for each Feature by Class:')
        print(df)
        return df

    def predict(self):
        self.stats = self.train_model(self.x_train, self.y_train)
        classes = np.log(self.y_train.value_counts() / len(self.y_train))
        predictions = pd.Series(np.zeros(len(np.unique(self.y_train))), index=np.unique(self.y_train))
        for i in range(len(self.x_test)):
            for val in np.unique(self.y_train):
                temp = 0
                for j in range(self.x_train.shape[1]):
                    temp += self.log_gaussian(self.x_test.iloc[i, j], self.stats.loc[(val, 'Mean'), j], self.stats.loc[(val, 'STD'), j])
                predictions[val] = classes[val] + temp
            self.y_hat[i] = np.argmax(predictions)
        accuracy = np.mean([1 if self.y_test.iloc[i] == self.y_hat[i] else 0 for i in range(self.y_test.shape[0])])
        print('\nNaive Bayes Accuracy:', accuracy)
        return accuracy
if __name__ == '__main__':
    RATIO = 0.75
    SEED = 123
    PATH = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
    classifier = NaiveBayes(PATH, RATIO, SEED)
    classifier.import_data()
    classifier.predict()