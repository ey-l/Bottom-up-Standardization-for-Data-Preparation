import pandas
data = pandas.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(data.shape)
data.head(10)
data.dtypes
data.describe()
data.groupby('Outcome').size()

import matplotlib.pyplot as plt
data.hist()
