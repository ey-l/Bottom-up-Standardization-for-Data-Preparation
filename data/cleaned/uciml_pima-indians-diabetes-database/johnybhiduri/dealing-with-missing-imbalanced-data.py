import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings(action='ignore')
pima_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
pima_df.head()
pima_df.iloc[:, 1:-1] = pima_df.iloc[:, 1:-1].replace(0, np.nan)
pima_df.isna().sum()

def SimpleImp(df, stra):
    df = df.copy()
    imputer = SimpleImputer(strategy=stra)
    df1 = imputer.fit_transform(df)
    df1 = pd.DataFrame(df1, columns=df.columns, index=df.index)
    (x_train, x_test, y_train, y_test) = train_test_split(df1.iloc[:, 0:-1], df1.iloc[:, -1], test_size=0.3, shuffle=True, random_state=1)
    model = LogisticRegression()