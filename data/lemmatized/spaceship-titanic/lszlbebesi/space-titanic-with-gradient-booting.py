import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from statsmodels.stats.diagnostic import het_white
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def non_zero_stuff(data):
    return np.count_nonzero(data.isnull())

def len_different(x):
    return len(list(set(x)))

def WOE_based_IV(df, target, cont_var, limits):
    woe_df = pd.DataFrame()
    bins_list = list()
    event_list = list()
    non_event_list = list()
    for i in range(1, len(limits)):
        even_count = np.nansum(df[(limits[i - 1] < df[cont_var]) & (df[cont_var] <= limits[i])][target] > 0)
        non_even_count = np.nansum(df[(limits[i - 1] < df[cont_var]) & (df[cont_var] <= limits[i])][target] < 1)
        event_list.append(even_count)
        non_event_list.append(non_even_count)
        bins_list.append('lower: ' + str(limits[i - 1]) + ' - upper: ' + str(limits[i]))
    woe_df = pd.DataFrame({'bin': bins_list, 'No_events': event_list, 'No_nonevents': non_event_list})
    woe_df['event_pct'] = woe_df['No_events'] / sum(woe_df['No_events'])
    woe_df['nonevent_pct'] = woe_df['No_nonevents'] / sum(woe_df['No_nonevents'])
    woe_df['WOE'] = np.log(woe_df['event_pct'] / woe_df['nonevent_pct'])
    woe_df['IV'] = (woe_df['event_pct'] - woe_df['nonevent_pct']) * woe_df['WOE']
    return woe_df

def fit_model_using_classifier(alg, dtrain, predictors, target='Transported', performCV=True, printFeatureImportance=True, cv_folds=3, repeat=5, scoring='roc_auc', only_top_x_feature=60):
    """
    I used the function found in this source
    https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    I modified the code slightly
    """
    cv_score = list()
    if performCV:
        for i in range(0, repeat):
            cv_score_temp = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=cv_folds, scoring=scoring)
            cv_score = cv_score + list(cv_score_temp)