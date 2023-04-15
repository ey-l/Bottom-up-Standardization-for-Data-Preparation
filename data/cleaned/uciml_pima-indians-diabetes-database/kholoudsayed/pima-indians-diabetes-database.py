import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math, time, random, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier, Pool, cv
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
train
train.info()
train.shape
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train.isnull().sum()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train.iloc[:, 1:6] = train.iloc[:, 1:6].replace(0, np.NaN)
train.dropna(thresh=2, axis=0, inplace=True)
train.shape
imputer = SimpleImputer(missing_values=np.NAN, strategy='mean')