import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
print(data.shape)
data.head(3)
data.isnull().values.any()
corrmat = data.corr()
top_corr_features = corrmat.index
pass
pass
data.columns = ['num_preg', 'glucose', 'blood_pres', 'Skinthickness', 'insulin', 'bmi', 'diab_pred', 'age', 'diabetes']
data.head(3)
diabetes_true_count = len(data.loc[data['diabetes'] == 1])
diabetes_false_count = len(data.loc[data['diabetes'] == 0])
print('1s :', diabetes_true_count, '  0s :', diabetes_false_count)
print('total number of rows : {0}'.format(len(data)))
for col in data.columns[:-1]:
    print('number of zeros ', col, '{0}'.format(len(data.loc[data[col] == 0])))
imputer = SimpleImputer(missing_values=0, strategy='mean')
X_train = imputer.fit_transform(data.loc[:, :'age'])
XX = data.loc[:, :'age'].values
YY = data.loc[:, 'diabetes'].values
(X_train, X_test, y_train, y_test) = train_test_split(XX, YY, test_size=0.2, random_state=0)
random_forest_model = RandomForestClassifier(random_state=0, max_features=5, n_estimators=500)