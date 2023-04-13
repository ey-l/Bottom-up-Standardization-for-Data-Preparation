import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import json
import sklearn
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from xgboost import plot_tree
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
font = {'family': 'Helvetica, Ariel', 'weight': 'normal', 'size': 12}
plt.rc('font', **font)
sns.set(rc={'figure.dpi': 300, 'savefig.dpi': 300})
sns.set_context('notebook')
sns.set_style('ticks')
FIG_FONT = dict(family='Helvetica, Ariel', weight='bold', color='#7f7f7f')
sns.set_palette('Spectral')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
df = _input1.copy()
df.head()
df.info()
df.describe()
import pandas_profiling
df.profile_report()
df = _input1.copy()
df = df.drop(['Name', 'PassengerId'], axis=1)
df = df.dropna()
target = df['Transported']
df = df.drop(['Transported'], axis=1)
target = target.astype(int)
df['Cabin_1'] = df['Cabin'].apply(lambda x: re.findall('[\\w+]', x)[0] if pd.isnull(x) == False else float('nan'))
df['Cabin_2'] = df['Cabin'].apply(lambda x: re.findall('[\\w]+', x)[1] if pd.isnull(x) == False else float('nan'))
df['Cabin_2'] = pd.cut(df['Cabin_2'].astype('int'), 5, labels=[0, 1, 2, 3, 4])
df['Cabin_3'] = df['Cabin'].apply(lambda x: re.findall('[\\w]+', x)[2] if pd.isnull(x) == False else float('nan'))
df = df.drop(['Cabin'], axis=1)
numaric_columns = list(df.select_dtypes(include=np.number).columns)
print('Numaric columns (' + str(len(numaric_columns)) + ') :', ', '.join(numaric_columns))
cat_columns = df.select_dtypes(include=['object']).columns.tolist()
print('Categorical columns (' + str(len(cat_columns)) + ') :', ', '.join(cat_columns))
from collections import Counter

def detect_outliers(df, n, features_list):
    outlier_indices = []
    for feature in features_list:
        Q1 = np.percentile(df[feature], 25)
        Q3 = np.percentile(df[feature], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[feature] < Q1 - outlier_step) | (df[feature] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list((key for (key, value) in outlier_indices.items() if value > n))
    return multiple_outliers
outliers_to_drop = detect_outliers(df, 2, numaric_columns)
print('We will drop these {} indices: '.format(len(outliers_to_drop)), outliers_to_drop)
(X_train, X_test, y_train, y_test) = train_test_split(df, target, test_size=0.2, random_state=100, stratify=target)
X_train_n = X_train[numaric_columns].copy()
X_test_n = X_test[numaric_columns].copy()
X_train_c = X_train[cat_columns]
X_test_c = X_test[cat_columns]
encoder = OrdinalEncoder()
X_train_c = encoder.fit_transform(X_train_c)
X_train_c = pd.DataFrame(X_train_c, index=X_train_n.index)
X_test_c = encoder.transform(X_test_c)
X_test_c = pd.DataFrame(X_test_c, index=X_test_n.index)
for (i, column) in enumerate(X_train_c.columns):
    X_train_n['cat_' + str(i + 1)] = X_train_c[column]
    X_test_n['cat_' + str(i + 1)] = X_test_c[column]
X_train = X_train_n
X_test = X_test_n
import scipy.stats as stats

def univariate_double_plot(df=df, x=None, xlabel=None, explode=None, ylabel=None, palette=None, order=True, hue=None):
    sns.set_palette(palette)
    (fig, ax) = plt.subplots(1, 2, figsize=(20, 7))
    if order == True:
        feature_data = df[x].value_counts(ascending=True)
        sns.countplot(data=df, x=x, ax=ax[0], order=feature_data.index, hue=hue)
    else:
        feature_data = df[x].value_counts(sort=False).sort_index()
        sns.countplot(data=df, x=x, ax=ax[0], order=feature_data.index, hue=hue)
    (patches, texts, autotexts) = ax[1].pie(feature_data.values, labels=feature_data.index, autopct='%.0f%%', textprops={'size': 20})
    for i in range(len(autotexts)):
        autotexts[i].set_color('white')
    sns.despine(bottom=True, left=True)
    for i in range(len(ax[0].containers)):
        ax[0].bar_label(ax[0].containers[i], label_type='edge', size=12, padding=1, fontname='Helvetica, Ariel', color='#7f7f7f')
    ax[0].set_xlabel(xlabel=xlabel, size=12, fontdict=FIG_FONT)
    ax[0].set_ylabel(ylabel=ylabel)
    ax[1].set_ylabel(ylabel=ylabel)
    fig.text(0.5, 1, f'{xlabel} Distribution', size=16, fontdict=FIG_FONT, ha='center', va='center')

def univariate_single_plot(df=df, x=None, xlabel=None, rotation=None, ylabel=None, palette=None):
    sns.set_palette(palette)
    (fig, ax) = plt.subplots(1, 1, figsize=(20, 7))
    feature_data = df[x].value_counts(ascending=True)
    sns.countplot(data=df, x=x, order=df[x].value_counts(ascending=True).index)
    sns.despine(bottom=True, left=True)
    plt.xlabel(xlabel=xlabel, size=14, fontdict=FIG_FONT)
    plt.xticks(rotation=rotation)
    plt.ylabel(ylabel=ylabel)
    for i in range(len(feature_data.index)):
        ax.text(i, feature_data.iloc[i] * 0.9, feature_data.iloc[i], ha='center', fontsize=20, color='white')
    plt.title(label=f'{xlabel} Distribution', size=18, fontdict=FIG_FONT)

def univariate_numerical_plot(df=df, x=None, xlabel=None, ylabel=None, palette=None, bins=20):
    sns.set_palette(palette)
    (fig, ax) = plt.subplots(1, 3, figsize=(20, 7))
    sns.histplot(bins=bins, data=df, x=x, kde=True, ax=ax[0])
    sns.boxplot(data=df, y=x, ax=ax[1])
    plt.sca(ax[2])
    stats.probplot(df[x], dist='norm', plot=plt)
    plt.ylabel('Variable quantiles')
    sns.despine(bottom=True, left=True)
    ax[0].set_xlabel(xlabel=xlabel, size=12, fontdict=FIG_FONT)
    ax[0].set_title(f'The histogram of {x}')
    ax[1].set_xlabel(xlabel=ylabel, size=12, fontdict=FIG_FONT)
    ax[0].set_ylabel(ylabel=ylabel, size=12, fontdict=FIG_FONT)
    ax[1].set_ylabel(ylabel=xlabel, size=12, fontdict=FIG_FONT)
    ax[1].set_title(f'The boxplot of {x}')
    fig.text(0.5, 1, f'{xlabel} Distribution', size=16, fontdict=FIG_FONT, ha='center', va='center')
feature_data = univariate_numerical_plot(df, 'Age')
x = 'HomePlanet'
univariate_double_plot(_input1, x, x, hue='Transported')
x = 'CryoSleep'
univariate_double_plot(_input1, x, x, hue='Transported')
x = 'Destination'
univariate_double_plot(_input1, x, x, hue='Transported')
x = 'VIP'
univariate_double_plot(_input1, x, x, hue='Transported')

def Chi_square_test(data, use_method=1, alpha=0.05):
    data = data.astype(int)
    data_j = data.sum() / data.sum().sum()
    data_i = data.sum(axis=1)
    k = (len(data.columns) - 1) * (len(data) - 1)
    rej_boundary = stats.chi2.ppf(1 - alpha / 2, k)
    data_exp = np.dot(data_i.values.reshape(-1, 1), data_j.values.reshape(1, -1))
    data_exp = pd.DataFrame(data_exp, index=data_i.index, columns=data_j.index)
    data_obj = data.values.flatten()
    data_exp = data_exp.values.flatten()

    def calculate_chi_val(use_method):
        if use_method == 1:
            (chi_val, p_val) = stats.chisquare(f_obs=data_obj, f_exp=data_exp, ddof=len(data_obj) - 1 - (len(data.columns) - 1) * (len(data) - 1))
            return (chi_val, p_val)
        elif use_method == 2:
            chi_val = 0
            for i in range(len(data_obj)):
                chi_val += (abs(data_obj[i] - data_exp[i]) - 0.5) ** 2 / data_exp[i]
            p_val = scipy.stats.chi2.sf(abs(chi_val), k)
            return (chi_val, p_val)
    (chi_val, p_val) = calculate_chi_val(use_method)
    print('The Chi square value of the sample is:{:.3f}, The corresponding degrees of freedom is: {}, \nwhen alpha={}, p-value is :{:.8f}, Reject domain boundary is: {:.3f}'.format(chi_val, k, alpha, p_val, rej_boundary))
    if chi_val > rej_boundary:
        print(f'Conclusion： alpha=0.05, refuse H0, accpet H1. There are essential differences between {data.index.values} in {data.index.name}')
        print('-' * 120)
    else:
        print(f'Conclusion: alpha=0.05, accpet H0, refuse H1. There is no essential difference between {data.index.values} in {data.index.name}')
        print('-' * 120)
categories = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for category in categories:
    feature = [category, 'Transported']
    data = _input1[feature]
    data = data.pivot_table(index=feature[0], columns=feature[1], aggfunc=len, fill_value=0)
    if 'No internet service' in data.index.values:
        data = data.drop('No internet service', inplace=False)
    if 'No phone service' in data.index.values:
        data = data.drop('No phone service', inplace=False)
    print(f'data matrix:\n{data}\n')
    Chi_square_test(data, len(data.index.values) - 1, 0.01)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
temp_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
PARAMETERS = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.5}
cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=10, params=PARAMETERS, as_pandas=True, seed=123)
cv_results
results = []
for value in range(3, 10, 1):
    PARAMETERS['max_depth'] = value
    cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=10, params=PARAMETERS, as_pandas=True, seed=123)
    results.append((cv_results['train-auc-mean'].tail().values[-1], cv_results['test-auc-mean'].tail().values[-1]))
data = list(zip(range(3, 10, 1), results))
print(pd.DataFrame(data, columns=['max_depth', 'auc(train,test)']))
results = []
for value in [0.1, 0.2, 0.5, 1, 1.5, 2]:
    PARAMETERS['gamma'] = value
    cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=10, params=PARAMETERS, as_pandas=True, seed=123)
    results.append((cv_results['train-auc-mean'].tail().values[-1], cv_results['test-auc-mean'].tail().values[-1]))
data = list(zip([0.1, 0.2, 0.5, 1, 1.5, 2], results))
print(pd.DataFrame(data, columns=['gamma', 'auc(train,test)']))
results = []
for value in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    PARAMETERS['colsample_bytree'] = value
    cv_results = xgb.cv(dtrain=temp_dmatrix, nfold=5, num_boost_round=10, params=PARAMETERS, as_pandas=True, seed=123)
    results.append((cv_results['train-auc-mean'].tail().values[-1], cv_results['test-auc-mean'].tail().values[-1]))
data = list(zip([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], results))
print(pd.DataFrame(data, columns=['colsample_bytree', 'auc(train,test)']))
learning_rate = [0.1, 0.3, 0.6]
max_depth = [3, 6, 8]
gamma = [0.2, 0.5, 1]
subsample = [0.9, 1]
colsample_bytree = [0.8, 1]
reg_lambda = np.linspace(0.001, 1, 2).tolist()
parameters = {'learning_rate': learning_rate, 'max_depth': max_depth, 'gamma': gamma, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'reg_lambda': reg_lambda}
model = xgb.XGBClassifier(tree_method='gpu_hist')
clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)