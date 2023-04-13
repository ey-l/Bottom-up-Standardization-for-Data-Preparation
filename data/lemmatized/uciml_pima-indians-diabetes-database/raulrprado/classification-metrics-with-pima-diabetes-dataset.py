import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pass
pass
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
data.isnull().sum()
features = list(data.columns)
features.remove('Outcome')
N_TRUE = len(data[data['Outcome'] == 1])
N_FALSE = len(data) - N_TRUE
print('N_TRUE = {}'.format(N_TRUE))
print('N_FALSE = {}'.format(N_FALSE))
print('N_FALSE fraction = {:.3f}'.format(N_FALSE / (N_FALSE + N_TRUE)))
pass
axs = axs.flatten()
for (ax, feat) in zip(axs, features):
    pass
data_with_zeros = data[(data['Glucose'] == 0) | (data['BMI'] == 0) | (data['Insulin'] == 0) | (data['BloodPressure'] == 0)]
print('N of examples with incomplete data = {}'.format(len(data_with_zeros)))
for feat in ['Glucose', 'BloodPressure', 'BMI', 'Insulin']:
    new_feat = 'Valid' + feat
    data[new_feat] = data[feat].map(lambda d: int(d != 0))
    features.append(new_feat)
print(features)
pass
pass
from sklearn.preprocessing import StandardScaler
(X_train, y_train) = (data[features].to_numpy(), data['Outcome'].to_numpy())