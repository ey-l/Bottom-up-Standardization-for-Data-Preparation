import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pass
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.sample(5)
data.info()
data.describe()
data.isnull().sum()
pass
pass
import warnings
warnings.filterwarnings('ignore')
pass
q = data['Insulin'].quantile(0.99)
data1 = data[data['Insulin'] < q]
pass
pass
bl = data['Outcome'].value_counts()
bl.plot(kind='bar', rot=360, color='red')
data1.plot(kind='kde', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(15, 10))
corr_metrics = data1.corr()
corr_metrics.style.background_gradient()
pass
pass
pass
data2 = data1[data1['BloodPressure'] != 0]
sum(data2['BloodPressure'] == 0)
X = data2.drop(columns=['Outcome'])
y = data2['Outcome']
pass
pass
ax[0].set_title('Before Oversampling')
pass
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
(X, y) = sm.fit_resample(X, y)
pass
ax[1].set_title('After Oversampling')
pass
pass
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=40)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
pipe1 = make_pipeline(MinMaxScaler(), SVC())
pipe2 = make_pipeline(Normalizer(), SVC())
pipe3 = make_pipeline(RobustScaler(), SVC())
pipe4 = make_pipeline(StandardScaler(), SVC())
parameter = {'svc__C': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200], 'svc__gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]}
grid = GridSearchCV(pipe1, parameter, cv=5)