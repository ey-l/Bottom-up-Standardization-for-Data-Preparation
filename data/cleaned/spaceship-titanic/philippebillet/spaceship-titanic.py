import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import re
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
from sklearn import set_config
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from pandas.api.types import is_numeric_dtype
from itertools import product
from joblib import dump
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from yellowbrick.classifier import classification_report
from yellowbrick.classifier import class_prediction_error
from yellowbrick.classifier import confusion_matrix
from yellowbrick.classifier import precision_recall_curve
from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.classifier.threshold import discrimination_threshold
from yellowbrick.features import pca_decomposition
from yellowbrick.features import rank1d
from yellowbrick.features import rank2d
from yellowbrick.target import balanced_binning_reference
from yellowbrick.target import class_balance
from yellowbrick.model_selection import learning_curve
from yellowbrick.model_selection import feature_importances
from yellowbrick.contrib.missing import MissingValuesBar
from yellowbrick.contrib.missing import MissingValuesDispersion
from yellowbrick.target.feature_correlation import feature_correlation
import ipywidgets as widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, is_classifier, is_regressor, TransformerMixin
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, recall_score

def plot_dataframe_structure(df):
    """
    Plot dataframe structure: It shows the different data types in the dataframe.

    Parameters
    ----------
    df: Pandas dataframe
    
    Returns
    -------
    Plotting
    """
    plt.figure()
    df.dtypes.value_counts().plot.pie(ylabel='')
    plt.title('Data types')


def plot_categorical(df):
    """
    Plot the number of different values for each categorical feature in the dataframe.

    Parameters
    ----------
    df: Pandas dataframe
    
    Returns
    -------
    Plotting
    """
    plt.figure()
    df.nunique().plot.bar()
    plt.title('Number of different values')


def duplicates(df):
    """
    Remove the duplicate rows from dataframe.

    Parameters
    ----------
    df: Pandas dataframe
    
    Returns
    -------
    df: Pandas dataframe without duplicate rows 
    """
    duplicate_rows_df = df[df.duplicated()]
    if duplicate_rows_df.shape[0] > 0:
        print('Number of rows before removing:', df.count()[0])
        print('Number of duplicate rows:', duplicate_rows_df.shape[0])
        df = df.drop_duplicates()
        print('Number of rows after removing:', df.count()[0])
    else:
        print('No duplicate rows.')
    return df

def drop_na(df, threshold_NaN):
    """
    Remove the columns from dataframe containing NaN depending on threshold_NaN.

    Parameters
    ----------
    df: Pandas dataframe
    threshold_NaN: in [0, 1], from GUI
    
    Returns
    -------
    df: Pandas dataframe 
    drop_cols: list of dropped columns
    """
    isna_stat = (df.isna().sum() / df.shape[0]).sort_values(ascending=True)
    drop_cols = []
    if isna_stat.max() > 0.0:
        drop_cols = np.array(isna_stat[isna_stat > threshold_NaN].index)
        print('Drop columns containing more than', threshold_NaN * 100, '% of NaN:', drop_cols)
        df = df.drop(drop_cols, axis=1)
    else:
        print('No need to drop columns.')
    return (df, drop_cols)

def encoding(df, threshold_cat, target_col):
    """
    Encode the data.

    Parameters
    ----------
    df: Pandas dataframe
    threshold_cat: integer, if the number of different values of a given column is less than this limit, 
    this column is considered as categorical. 
    
    Returns
    -------
    df: Pandas dataframe 
    encoded_cols: Pandas dataframe of columns with their encoding and range
    """
    encoded_cols = []
    for c in df.columns:
        if df[c].dtypes == 'object' or df[c].dtypes.name == 'category':
            encoded_cols.append([c, 'cat', df[c].dropna().unique().tolist()])
            print('Encoding object column:', c)
            df[c] = df[c].factorize()[0].astype(np.int)
        elif is_numeric_dtype(df[c]):
            if df[c].unique().shape[0] > threshold_cat:
                encoded_cols.append([c, 'num', [df[c].min(), df[c].max()]])
                print('Encoding numeric column:', c)
                df[c] = (df[c] - df[c].mean()) / df[c].std()
            else:
                print('Column ', c, ' is categorical.')
                encoded_cols.append([c, 'cat', df[c].dropna().unique().tolist()])
        else:
            print('Unknown type ', df[c].dtypes, ' for column:', c)
            df = df.drop(c, axis=1)
            drop_cols = np.unique(np.concatenate((drop_cols, c)))
    encoded_cols = pd.DataFrame(np.array(encoded_cols), columns=['column_name', 'column_type', 'column_range'])
    encoded_cols = encoded_cols.loc[encoded_cols['column_name'] != target_col]

    return (df, encoded_cols)

def imputation(df):
    """
    Impute NaN in the dataframe using IterativeImputer.

    Parameters
    ----------
    df: Pandas dataframe
    
    Returns
    -------
    df: Pandas dataframe 
    """
    isna_stat = (df.isna().sum() / df.shape[0]).sort_values(ascending=True)
    if isna_stat.max() > 0.0:
        print('Imputing NaN using IterativeImputer')
        df = pd.DataFrame(IterativeImputer(random_state=0).fit_transform(df), columns=df.columns)
    else:
        print('No need to impute data.')
    return df

def outliers(df, threshold_Z):
    """
    Remove the outliers from dataframe according to Z_score.

    Parameters
    ----------
    df: Pandas dataframe
    threshold_Z: number from GUI. 
    
    Returns
    -------
    df: Pandas dataframe. 
    """
    Z_score = np.abs(stats.zscore(df))
    df_o_Z = df[(Z_score < threshold_Z).all(axis=1)]
    if df_o_Z.shape[0] != 0:
        print('Using Z_score, ', str(df.shape[0] - df_o_Z.shape[0]), ' rows will be suppressed.')
        df = df_o_Z
    else:
        print('Possible problem with outliers treatment, check threshold_Z')
    return df

def correlated_columns(df, threshold_corr, target_col):
    """
    Display correlation matrix of features, and returns the list of the too correlated features
    according to threshold_corr.

    Parameters
    ----------
    df: Pandas dataframe
    threshold_corr: number from GUI
    target: target column
    Returns
    -------
    correlated_features: list of the features having a correlation greater than threshold_corr. 
    """
    df = df.drop(target_col, axis=1)
    corr_matrix = df.corr()
    correlated_features = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold_corr:
                colname = corr_matrix.columns[i]
                correlated_features.append(colname)
    correlated_features = list(dict.fromkeys(correlated_features))
    return correlated_features

def plot_sns_corr_class(df, target_col):
    """
    Plot correlation information for classification problem (if Seaborn option is checked).

    Parameters
    ----------
    df: Pandas dataframe
    target_col: name of the target column. 
    
    Returns
    -------
    Plotting. 
    """
    g = sns.PairGrid(df, hue=target_col)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    g.add_legend()
    g.fig.suptitle('Pairwise data relationships', y=1.01)


def plot_sns_corr_regre(df, target_col):
    """
    Plot correlation information for regression problem (if Seaborn option is checked).

    Parameters
    ----------
    df: Pandas dataframe
    target_col: name of the target column. 
    
    Returns
    -------
    Plotting. 
    """
    g = sns.PairGrid(df)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    g.fig.suptitle('Pairwise data relationships', y=1.01)


class Decorrelator(BaseEstimator, TransformerMixin):
    """
    Decorrelator is a class used to eliminate too correlated columns depending on a threshold during preprocessing.

    Parameters
    ----------
    threshold_corr
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.correlated_columns = None

    def fit(self, X, y=None):
        correlated_features = set()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)
        self.correlated_features = correlated_features
        return self

    def transform(self, X, y=None, **kwargs):
        return pd.DataFrame(X).drop(labels=self.correlated_features, axis=1)

class ColumnsDropper(BaseEstimator, TransformerMixin):
    """
    ColumnsDropper is a class used to drop columns from a dataset.

    Parameters
    ----------
    cols : list of columns dropped by the transformer
    """

    def __init__(self, cols):
        if not isinstance(cols, list):
            self.cols = [cols]
        else:
            self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        return X[self.cols]

def model_filtering(level_0, model_imp, nb_model, score_stack, threshold_score):
    """
    Suppress estimators from level 0 having a test score smaller than threshold_score (from score_stack), then 
    keep nb_model best estimators (according to model_imp).
    Parameters
    ----------
    level_0: list of estimators of level 0
    model_imp: sorted array of model importance
    nb_model : number of model to keep
    score_stack: accuracy of estimators on train and test sets in a tabular
    threshold_score : minimal score
    
    Returns
    -------
    list of filtered estimators of level 0
    """
    if nb_model > len(level_0):
        nb_model = len(level_0)
    score_stack = np.delete(np.delete(score_stack, 1, axis=1), -1, axis=0)
    score_stack = score_stack[score_stack[:, 1] > threshold_score]
    if nb_model > len(score_stack):
        nb_model = len(score_stack)
    model_imp = model_imp[np.in1d(model_imp[:, 0], score_stack)]
    model_imp_f = model_imp[np.argpartition(model_imp[:, 1], -nb_model)[-nb_model:]].T[0]
    return list(filter(lambda x: x[0] in model_imp_f, level_0))

def feature_filtering(feature_importance, nb_feature):
    """
    Separate features in two lists, the first one contains the nb_feature most important features, 
    the second one contains the complement
    Parameters
    ----------
    feature_importance: array of features with their importance
    nb_feature: number of features we want to keep
    
    Returns
    -------
    best_feature: list of nb_feature most important features
    worst_feature: list of the worst important features
    """
    if nb_feature > feature_importance.shape[0]:
        nb_feature = feature_importance.shape[0]
    best_feature = feature_importance[np.argpartition(feature_importance[:, 1], -nb_feature)[-nb_feature:]].T[0]
    worst_feature = list(set(feature_importance.T[0]) - set(best_feature))
    return (best_feature, worst_feature)

def split(X, y, test_size=0.33, threshold_entropy=0.7, undersampling=False, undersampler=None):
    """
    Split dataframe into train and test sets.
    If the Shannon entropy of the target dataset is less than 0.7, RepeatedStratifiedKFold is used

    Parameters
    ----------
    X: feature dataframe
    y: target dataframe
    
    Returns
    -------
    X_train: train feature dataframe 
    X_test: test feature dataframe
    y_train: train target dataframe
    y_test: test target dataframe
    """
    s_e = shannon_entropy(y)
    if s_e < threshold_entropy:
        if undersampling:
            if undersampler == 'Random':
                from imblearn.under_sampling import RandomUnderSampler
                us = RandomUnderSampler()
            elif undersampler == 'Centroids':
                from imblearn.under_sampling import ClusterCentroids
                us = ClusterCentroids()
            elif undersampler == 'AllKNN':
                from imblearn.under_sampling import AllKNN
                us = AllKNN()
            elif undersampler == 'TomekLinks':
                from imblearn.under_sampling import TomekLinks
                us = TomekLinks()
            else:
                print('Unknown undersampler')
            (X, y) = us.fit_resample(X, y)
            print('Shannon Entropy = {:.4}, split using undersampler {} and RepeatedStratifiedKFold'.format(s_e, undersampler))
        else:
            print('Shannon Entropy = {:.4}, split using RepeatedStratifiedKFold'.format(s_e))
        skfold = RepeatedStratifiedKFold(n_splits=5)
        for (ind_train, ind_test) in skfold.split(X, y):
            (X_train, X_test) = (X.iloc[ind_train], X.iloc[ind_test])
            (y_train, y_test) = (y.iloc[ind_train], y.iloc[ind_test])
    else:
        (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, stratify=None, shuffle=True)
    return (X_train, X_test, y_train, y_test)

def downcast_dtypes(df):
    """
    Compress dataframe

    Parameters
    ----------
    df: Pandas dataframe
    
    Returns
    -------
    df: Pandas dataframe
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f}MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f}MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def shannon_entropy(y):
    """
    Compute Shannon entropy of a dataset

    Parameters
    ----------
    y: univariate Pandas dataframe
    
    Returns
    -------
    shannon entropy: float
    """
    from collections import Counter
    from numpy import log
    n = len(y)
    classes = [(clas, float(count)) for (clas, count) in Counter(y).items()]
    k = len(classes)
    H = -sum([count / n * log(count / n) for (clas, count) in classes])
    return H / log(k)

def score_stacking_c(model, X_train, y_train, X_test, y_test):
    """
    Compute the score of the stacked classification estimator and of each level_0 estimator

    Parameters
    ----------
    model: estimator obtained after fitting
    X_train: train feature dataframe 
    X_test: test feature dataframe
    y_train: train target dataframe
    y_test: test target dataframe
    
    Returns
    -------
    plotting: accuracy of estimators on train and test sets
    res_stack: accuracy of estimators on train and test sets in a tabular
    """
    nb_estimators = len(model.estimators_)
    res_stack = np.empty((nb_estimators + 1, 3), dtype='object')
    m_t_x_train = model.transform(X_train)
    for j in range(nb_estimators):
        res_stack[j, 0] = [*model.named_estimators_.keys()][j]
        if m_t_x_train.shape[1] == nb_estimators:
            res_stack[j, 1] = accuracy_score(np.rint(m_t_x_train).T[j], y_train)
            res_stack[j, 2] = accuracy_score(np.rint(model.transform(X_test)).T[j], y_test)
        else:
            res_stack[j, 1] = accuracy_score(m_t_x_train.reshape((X_train.shape[0], nb_estimators, y_train.unique().shape[0])).argmax(axis=2).T[j], y_train)
            res_stack[j, 2] = accuracy_score(model.transform(X_test).reshape((X_test.shape[0], nb_estimators, y_test.unique().shape[0])).argmax(axis=2).T[j], y_test)
    res_stack[len(model.estimators_), 0] = 'Stack'
    res_stack[len(model.estimators_), 1] = accuracy_score(model.predict(X_train), y_train)
    res_stack[len(model.estimators_), 2] = accuracy_score(model.predict(X_test), y_test)
    models = res_stack.T[0]
    score_train = res_stack.T[1]
    score_test = res_stack.T[2]
    plt.figure(figsize=(8, 5))
    plt.scatter(models, score_train, label='Train')
    plt.scatter(models, score_test, label='Test')
    plt.title('Model scores: accuracy')
    plt.xticks(rotation='vertical')
    plt.legend()

    return res_stack

def score_stacking_r(model, X_train, y_train, X_test, y_test):
    """
    Compute the score of the stacked regression estimator and of each level_0 estimator

    Parameters
    ----------
    model: estimator obtained after fitting
    X_train: train feature dataframe 
    X_test: test feature dataframe
    y_train: train target dataframe
    y_test: test target dataframe
    
    Returns
    -------
    plotting: accuracy of estimators on train and test sets
    res_stack: accuracy of estimators on train and test sets in a tabular
    """
    nb_estimators = len(model.estimators_)
    res_stack = np.empty((nb_estimators + 1, 3), dtype='object')
    m_t_x_train = model.transform(X_train)
    for j in range(nb_estimators):
        res_stack[j, 0] = [*model.named_estimators_.keys()][j]
        res_stack[j, 1] = r2_score(np.rint(m_t_x_train).T[j], y_train)
        res_stack[j, 2] = r2_score(np.rint(model.transform(X_test)).T[j], y_test)
    res_stack[len(model.estimators_), 0] = 'Stack'
    res_stack[len(model.estimators_), 1] = r2_score(model.predict(X_train), y_train)
    res_stack[len(model.estimators_), 2] = r2_score(model.predict(X_test), y_test)
    models = res_stack.T[0]
    score_train = res_stack.T[1]
    score_test = res_stack.T[2]
    plt.figure(figsize=(8, 5))
    plt.scatter(models, score_train, label='Train')
    plt.scatter(models, score_test, label='Test')
    plt.title('Model scores: r2')
    plt.xticks(rotation='vertical')
    plt.legend()

    return res_stack

def score_stacking(model, X_train, y_train, X_test, y_test):
    """
    Compute the score of the stacked estimator and of each level_0 estimator

    Parameters
    ----------
    model: estimator obtained after fitting
    X_train: train feature dataframe 
    X_test: test feature dataframe
    y_train: train target dataframe
    y_test: test target dataframe
    
    Returns
    -------
    plotting: accuracy of estimators on train and test sets
    res_stack: accuracy of estimators on train and test sets in a tabular
    plotting: model importance according to performance
    mod_imp: model importance in a tabular
    """
    if is_classifier(model):
        res_stack = score_stacking_c(model, X_train, y_train, X_test, y_test)
    else:
        res_stack = score_stacking_r(model, X_train, y_train, X_test, y_test)
    nb_estimators = len(model.estimators_)
    res_level_0 = res_stack[0:nb_estimators]
    mod_imp = np.delete(res_level_0[res_level_0[:, 2].argsort()], 1, axis=1)
    mod_imp.T[1] = mod_imp.T[1] / np.sum(mod_imp.T[1])
    (fig, ax) = plt.subplots()
    ax.barh(mod_imp.T[0], mod_imp.T[1])
    ax.set_title('Model Importance according to performance')
    fig.tight_layout()

    return (res_stack, mod_imp)

def model_importance_c(model):
    """
    Compute the model importance depending on final estimator coefficients for classification

    Parameters
    ----------
    model: estimator obtained after fitting

    Returns
    -------
    mod_imp: sorted array of model importance 
    """
    level_0 = np.array(list(model.named_estimators_.keys()))
    n_classes = model.classes_.shape[0]
    n_models = len(model.estimators_)
    if model.final_estimator_.coef_.shape[0] > 1:
        coeff = sum((model.final_estimator_.coef_.reshape(n_classes, n_models, n_classes)[i].T[i] for i in range(n_classes)))
    else:
        coeff = model.final_estimator_.coef_.reshape(n_models)
    model_importance = np.empty((len(level_0), 2), dtype='object')
    for ind in range(len(level_0)):
        model_importance[ind, 0] = level_0[ind]
        model_importance[ind, 1] = np.abs(coeff[ind])
    return model_importance[model_importance[:, 1].argsort()]

def model_importance_r(model):
    """
    Compute the model importance depending on final estimator coefficients for regression

    Parameters
    ----------
    model: estimator obtained after fitting
    
    Returns
    -------
    mod_imp: sorted array of model importance 
    """
    level_0 = np.array(list(model.named_estimators_.keys()))
    coeff = model.final_estimator_.coef_
    model_importance = np.empty((len(level_0), 2), dtype='object')
    for ind in range(len(level_0)):
        model_importance[ind, 0] = level_0[ind]
        model_importance[ind, 1] = np.abs(coeff[ind])
    return model_importance[model_importance[:, 1].argsort()]

def plot_model_importance(model):
    """
    Compute the model importance depending on final estimator coefficients

    Parameters
    ----------
    model: estimator obtained after fitting
    
    Returns
    -------
    plotting: model importance according to aggragator coefficients
    mod_imp: sorted array of model importance 
    """
    if is_classifier(model):
        mod_imp = model_importance_c(model)
    else:
        mod_imp = model_importance_r(model)
    mod_imp.T[1] = mod_imp.T[1] / np.sum(mod_imp.T[1])
    (fig, ax) = plt.subplots()
    ax.barh(mod_imp.T[0], mod_imp.T[1])
    ax.set_title('Model Importance according to aggragator coefficients')
    fig.tight_layout()

    return mod_imp

def plot_perm_importance(model, X, y):
    """
    Compute the feature permutation importance

    Parameters
    ----------
    model: estimator obtained after fitting
    X: feature dataframe
    y: target dataframe
    
    Returns
    -------
    plotting: feature permutation importance
    perm_imp: sorted array of feature permutation importance
    """
    if is_classifier(model):
        scoring = 'accuracy'
    else:
        scoring = 'r2'
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=10, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    perm_imp = np.array([X.columns[sorted_idx], result.importances[sorted_idx].mean(axis=1).T]).T
    perm_imp.T[1] = perm_imp.T[1] / np.sum(perm_imp.T[1])
    (fig, ax) = plt.subplots()
    ax.barh(perm_imp.T[0], perm_imp.T[1])
    ax.set_title('Permutation Importance')
    fig.tight_layout()

    return perm_imp

def plot_partial_dependence_c(model, X, features):
    """
    Plot partial dependence of features for a given classification estimator and a given dataset

    Parameters
    ----------
    model: estimator obtained after fitting
    X: feature dataframe
    features: list of features
    
    Returns
    -------
    plotting: partial dependence of input features
    """
    target = model.classes_
    for ind in range(len(target)):
        (fig, ax) = plt.subplots(figsize=(16, 8))
        display = PartialDependenceDisplay.from_estimator(estimator=model, X=X, features=features, target=target[ind], n_cols=2, kind='both', subsample=50, n_jobs=-1, grid_resolution=20, ice_lines_kw={'color': 'tab:blue', 'alpha': 0.2, 'linewidth': 0.5}, pd_line_kw={'color': 'tab:orange', 'linestyle': '--'}, ax=ax)
        display.figure_.suptitle('Partial dependence for class ' + str(target[ind]))
        display.figure_.subplots_adjust(hspace=0.3)


def plot_partial_dependence_r(model, X, features):
    """
    Plot partial dependence of features for a given regression estimator and a given dataset

    Parameters
    ----------
    model: estimator obtained after fitting
    X: feature dataframe
    features: list of features
    
    Returns
    -------
    plotting: partial dependence of input features
    """
    (fig, ax) = plt.subplots(figsize=(16, 8))
    display = PartialDependenceDisplay.from_estimator(estimator=model, X=X, features=features, n_cols=2, kind='both', subsample=50, n_jobs=-1, grid_resolution=20, ice_lines_kw={'color': 'tab:blue', 'alpha': 0.2, 'linewidth': 0.5}, pd_line_kw={'color': 'tab:orange', 'linestyle': '--'}, ax=ax)
    display.figure_.suptitle('Partial dependence')
    display.figure_.subplots_adjust(hspace=0.3)


def plot_partial_dependence(model, X, features):
    """
    Plot partial dependence of features for a given estimator and a given dataset

    Parameters
    ----------
    model: estimator obtained after fitting
    X: feature dataframe
    features: list of features, if features = [], partial dependences will be plot for all numeric features
    
    Returns
    -------
    plotting: partial dependence of input features
    """
    if features == []:
        features = X.select_dtypes([np.number]).columns.tolist()
    else:
        features = np.intersect1d(features, X.select_dtypes([np.number]).columns.tolist()).tolist()
    if features == []:
        return 'No numeric feature'
    elif is_classifier(model):
        plot_partial_dependence_c(model, X, features)
    else:
        plot_partial_dependence_r(model, X, features)

def plot_history(history):
    """
    Plot learning curves of Keras neural network

    Parameters
    ----------
    history: history of Keras neural network
    
    Returns
    -------
    plotting: learning curves of Keras neural network
    """
    pd.DataFrame(history.history).plot(figsize=(12, 9))
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()


def K_confusion_matrix(model, X_train, y_train, X_test, y_test):
    """
    Plot confusion matrix of a classification estimator on train and test sets

    Parameters
    ----------
    model: estimator obtained after fitting
    X_train: train feature dataframe 
    X_test: test feature dataframe
    y_train: train target dataframe
    y_test: test target dataframe
    
    Returns
    -------
    plotting: confusion matrix on train and test sets
    """
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_train)
    if len(y_pred.shape) > 1:
        y_pred = np.around(y_pred).astype(int)
        y_pred = np.argmax(y_pred, axis=1)
        y_train = y_train.idxmax(axis=1)
    cm = confusion_matrix(y_train, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix on train set')

    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1:
        y_pred = np.around(y_pred).astype(int)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = y_test.idxmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion matrix on test set')


def K_classification_report(model, X_train, y_train, X_test, y_test):
    """
    Plot classification report of a classification estimator on train and test sets

    Parameters
    ----------
    model: estimator obtained after fitting
    X_train: train feature dataframe 
    X_test: test feature dataframe
    y_train: train target dataframe
    y_test: test target dataframe
    
    Returns
    -------
    plotting: classification report on train and test sets
    """
    y_pred = model.predict(X_train)
    if len(y_pred.shape) > 1:
        y_pred = np.around(y_pred).astype(int)
        y_pred = np.argmax(y_pred, axis=1)
        y_train = y_train.idxmax(axis=1)
    cr = classification_report(y_train, y_pred, output_dict=True)

    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1:
        y_pred = np.around(y_pred).astype(int)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = y_test.idxmax(axis=1)
    cr = classification_report(y_test, y_pred, output_dict=True)


def K_r2(model, X_train, y_train, X_test, y_test):
    """
    Compute R^2 of a regression estimator on train and test sets.

    Parameters
    ----------
    model: estimator obtained after fitting
    X_train: train feature dataframe 
    X_test: test feature dataframe
    y_train: train target dataframe
    y_test: test target dataframe
    
    Returns
    -------
    array: scores on train and test sets
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    dr2 = {'train': [r2_score(y_train, y_pred_train)], 'test': [r2_score(y_test, y_pred_test)]}

problem_type = 'classification'
data_size = 'large'
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
target_col = 'Transported'
threshold_NaN = 0.5
threshold_cat = 5
threshold_Z = 3.0
test_size = 0.3
threshold_entropy = 0.75
undersampling = False
undersampler = 'Random'
threshold_corr = 0.95
threshold_model = 5
threshold_score = 0.7
threshold_feature = 5
df[['Cabin_1', 'Cabin_2', 'Cabin_3']] = df['Cabin'].str.split('/', 2, expand=True)
df['Whole_spent'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
user_drop_cols = ['PassengerId', 'Name', 'VIP', 'Cabin']

df = df.drop(user_drop_cols, axis=1)

df_copy = df.copy()
df.shape

plot_dataframe_structure(df)

plot_categorical(df)
duplicates(df)
(df, drop_cols) = drop_na(df, threshold_NaN)
dropped_cols = np.unique(np.concatenate((drop_cols, user_drop_cols)))

(df, encoded_cols) = encoding(df, threshold_cat, target_col)
visualizer = MissingValuesBar(features=df.select_dtypes(include=np.number).columns.tolist())