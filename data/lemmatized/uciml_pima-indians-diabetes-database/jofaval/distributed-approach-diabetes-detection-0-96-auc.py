RANDOM_SEED = 42
import pandas as pd
import numpy as np
np.random.seed(RANDOM_SEED)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
dataframe = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataframe.shape
target = 'Outcome'
features = [col for col in dataframe.columns if col not in [target]]

def readable(col: str) -> str:
    if col.isupper():
        return col
    title = col[0]
    for character in col[1:]:
        if character.isupper():
            title += ' '
        title += character.lower()
    return title
_ = [print(readable(col)) for col in dataframe.columns]
dataframe.info()
dataframe.head(1)
dataframe.describe().transpose()
USE_AGE_FEATURES = True
if USE_AGE_FEATURES:
    dataframe['YoungAdult'] = np.where(dataframe['Age'] < 25, 1, 0)
    dataframe['Adult'] = np.where((dataframe['Age'] >= 25) & (dataframe['Age'] < 40), 1, 0)
    dataframe['Senior'] = np.where(dataframe['Age'] >= 40, 1, 0)
USE_VINCENT_FEATURES = True
if USE_VINCENT_FEATURES:
    dataframe.loc[:, 'N1'] = 0
    dataframe.loc[(dataframe['Age'] <= 30) & (dataframe['Glucose'] <= 120), 'N1'] = 1
    dataframe.loc[:, 'N2'] = 0
    dataframe.loc[dataframe['BMI'] <= 30, 'N2'] = 1
    dataframe.loc[:, 'N3'] = 0
    dataframe.loc[(dataframe['Age'] <= 30) & (dataframe['Pregnancies'] <= 6), 'N3'] = 1
    dataframe.loc[:, 'N4'] = 0
    dataframe.loc[(dataframe['Glucose'] <= 105) & (dataframe['BloodPressure'] <= 80), 'N4'] = 1
    dataframe.loc[:, 'N5'] = 0
    dataframe.loc[dataframe['SkinThickness'] <= 20, 'N5'] = 1
    dataframe.loc[:, 'N6'] = 0
    dataframe.loc[(dataframe['BMI'] < 30) & (dataframe['SkinThickness'] <= 20), 'N6'] = 1
    dataframe.loc[:, 'N7'] = 0
    dataframe.loc[(dataframe['Glucose'] <= 105) & (dataframe['BMI'] <= 30), 'N7'] = 1
    dataframe.loc[:, 'N9'] = 0
    dataframe.loc[dataframe['Insulin'] < 200, 'N9'] = 1
    dataframe.loc[:, 'N10'] = 0
    dataframe.loc[dataframe['BloodPressure'] < 80, 'N10'] = 1
    dataframe.loc[:, 'N11'] = 0
    dataframe.loc[(dataframe['Pregnancies'] < 4) & (dataframe['Pregnancies'] != 0), 'N11'] = 1
    dataframe['N0'] = dataframe['BMI'] * dataframe['SkinThickness']
    dataframe['N8'] = dataframe['Pregnancies'] / dataframe['Age']
    dataframe['N13'] = dataframe['Glucose'] / dataframe['DiabetesPedigreeFunction']
    dataframe['N12'] = dataframe['Age'] * dataframe['DiabetesPedigreeFunction']
pass
pass
pass
pass
pass
pass
observation_group = dataframe[dataframe['Age'] <= 50]
pass
pass
axes[0].legend(['Diabetic', 'Non-Diabetic'])
pass
axes[1].legend(['Diabetic', 'Non-Diabetic'])
columns = dataframe.columns
n_columns = len(columns)
pass
fig.suptitle('Distribution of the features')
pass
for (index, col) in enumerate(columns):
    ax = axes[int(index % n_columns)]
    ax.grid()
    pass
    pass
REMOVE_OUTLIERS = True
USE_SCIPY_HANDLING = True
USE_FILTER_HANDLING = True
if REMOVE_OUTLIERS and USE_SCIPY_HANDLING:
    import scipy
    z_scores = scipy.stats.zscore(dataframe)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    dataframe = dataframe[filtered_entries]
if REMOVE_OUTLIERS and USE_FILTER_HANDLING:
    dataframe = dataframe[dataframe['Glucose'] > 0]
    dataframe = dataframe[dataframe['BloodPressure'] > 0]
    dataframe = dataframe[dataframe['BMI'] > 0]
columns = dataframe.columns
n_columns = len(columns)
pass
fig.suptitle('Distribution of the features')
pass
for (index, col) in enumerate(columns):
    ax = axes[int(index % n_columns)]
    ax.grid()
    pass
    pass
correlation = dataframe.corr().abs()
matrix = np.triu(correlation)
pass
pass
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, PolynomialFeatures
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
target = 'Outcome'
features = [col for col in dataframe.columns if col not in [target]]
X = dataframe[features]
y = dataframe[target]
pass
fig.suptitle('Distribution and balancement of the outcome variable')
pass
axes[0].legend(['Non-diabetic', 'Diabetic'])
y.value_counts().plot.pie(ax=axes[1])
axes[1].legend(['Non-diabetic', 'Diabetic'])
UNDER_SAMPLE = False
OVER_SAMPLE = True
if UNDER_SAMPLE:
    from imblearn.under_sampling import RandomUnderSampler
    (X, y) = RandomUnderSampler(random_state=RANDOM_SEED).fit_resample(X, y)
if OVER_SAMPLE:
    from imblearn.over_sampling import RandomOverSampler
    (X, y) = RandomOverSampler(random_state=RANDOM_SEED).fit_resample(X, y)
pass
fig.suptitle('Distribution and balancement of the outcome variable, after the balancement')
pass
y.value_counts().plot.pie(ax=axes[1])
axes[1].legend(['Non-diabetic', 'Diabetic'])
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
X_min = X_train.min()
X_max = X_train.max()
X_train_norm = (X_train - X_min) / (X_max - X_min)
X_train_norm = X_train_norm.fillna(0)
X_test_norm = (X_test - X_min) / (X_max - X_min)
X_test_norm = X_test_norm.fillna(0)
default_target_names = ['Non-Diabetic', 'Diabetic']

def cmatrix(y_test, y_pred, cmap: str='Blues', target_names: List[str]=default_target_names, title: str='Example', figsize: Tuple[int, int]=(20, 10)) -> np.ndarray:
    df_cm = confusion_matrix(y_test, y_pred)
    df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]
    pass
    pass

def auc(model: Pipeline=None, data: pd.DataFrame=X_test, y_true: np.ndarray=y_test, probs: np.ndarray=None) -> float:
    assert model is not None or probs is not None
    if probs is None:
        probs = model.predict_proba(data)[:, 1]
    return roc_auc_score(y_true, probs)
from sklearn.linear_model import LogisticRegression
clf_log = make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), LogisticRegression(random_state=RANDOM_SEED))