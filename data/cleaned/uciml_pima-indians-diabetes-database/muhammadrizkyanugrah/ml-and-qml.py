
from sklearn.datasets import load_iris
iris_data = load_iris()
print(iris_data.DESCR)
features = iris_data.data
labels = iris_data.target
print('Features =\n', features)
print('labels =\n', labels)
from sklearn.preprocessing import MinMaxScaler
features = MinMaxScaler().fit_transform(features)
import pandas as pd
import seaborn as sns
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['class'] = pd.Series(iris_data.target)
sns.pairplot(df, hue='class', palette='tab10')
from sklearn.model_selection import train_test_split
from qiskit.utils import algorithm_globals
algorithm_globals.random_seed = 123
(train_features, test_features, train_labels, test_labels) = train_test_split(features, labels, train_size=0.8, random_state=algorithm_globals.random_seed)
from sklearn.svm import SVC
svc = SVC()