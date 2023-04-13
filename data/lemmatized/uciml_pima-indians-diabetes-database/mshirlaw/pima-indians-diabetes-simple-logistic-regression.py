import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
correlations = data.corr()
correlations['Outcome'].sort_values(ascending=False)

def visualise(data):
    pass
    ax.scatter(data.iloc[:, 1].values, data.iloc[:, 5].values)
    ax.set_title('Highly Correlated Features')
    pass
    pass
visualise(data)
data[['Glucose', 'BMI']] = data[['Glucose', 'BMI']].replace(0, np.NaN)
data.dropna(inplace=True)
visualise(data)
X = data[['Glucose', 'BMI']].values
y = data[['Outcome']].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
mean = np.mean(X, axis=0)
print('Mean: (%d, %d)' % (mean[0], mean[1]))
standard_deviation = np.std(X, axis=0)
print('Standard deviation: (%d, %d)' % (standard_deviation[0], standard_deviation[1]))
print(X[0:10, :])
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()