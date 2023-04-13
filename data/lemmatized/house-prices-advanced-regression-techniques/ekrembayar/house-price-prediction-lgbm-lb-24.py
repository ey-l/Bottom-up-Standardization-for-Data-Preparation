import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

def grab_col_names(dataframe, cat_th=10, car_th=20, show_date=False):
    date_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'datetime64[ns]']
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != 'O' and (dataframe[col].dtypes != 'datetime64[ns]')]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'date_cols: {len(date_cols)}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    if show_date == True:
        return (date_cols, cat_cols, cat_but_car, num_cols, num_but_cat)
    else:
        return (cat_cols, cat_but_car, num_cols, num_but_cat)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].nunique() >= 5 and temp_df[col].nunique() <= 20 and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        return temp_df

def missing_values(data, plot=False, target='SalePrice'):
    mst = pd.DataFrame({'Num_Missing': data.isnull().sum(), 'Missing_Ratio': data.isnull().sum() / data.shape[0]}).sort_values('Num_Missing', ascending=False)
    mst['DataTypes'] = data[mst.index].dtypes.values
    mst = mst[mst.Num_Missing > 0].reset_index().rename({'index': 'Feature'}, axis=1)
    mst = mst[mst.Feature != target]
    print('Number of Variables include Missing Values:', mst.shape[0], '\n')
    if mst[mst.Missing_Ratio >= 1.0].shape[0] > 0:
        print('Full Missing Variables:', mst[mst.Missing_Ratio >= 1.0].Feature.tolist())
        data = data.drop(mst[mst.Missing_Ratio >= 1.0].Feature.tolist(), axis=1, inplace=False)
        print('Full missing variables are deleted!', '\n')
    if plot:
        plt.figure(figsize=(25, 8))
        p = sns.barplot(mst.Feature, mst.Missing_Ratio)
        for rotate in p.get_xticklabels():
            rotate.set_rotation(90)
    print(mst, '\n')

def ordinal(serie, category):
    numeric = np.arange(1, len(category) + 1, 1)
    zip_iterator = zip(category, numeric)
    mapping = dict(zip_iterator)
    serie = serie.map(mapping)
    return serie

def transform_ordinal(data, ordinal_vars, category):
    for i in ordinal_vars:
        data[i] = ordinal(data[i], category=category)

def cat_analyzer(dataframe, variable, target):
    print(variable)
    print(pd.DataFrame({'COUNT': dataframe[variable].value_counts(), 'RATIO': dataframe[variable].value_counts() / len(dataframe), 'TARGET_COUNT': dataframe.groupby(variable)[target].count(), 'TARGET_MEAN': dataframe.groupby(variable)[target].mean(), 'TARGET_MEDIAN': dataframe.groupby(variable)[target].median(), 'TARGET_STD': dataframe.groupby(variable)[target].std()}), end='\n\n\n')

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def cart_feature_gen(model_type, dataframe, X, y, suffix=None):
    if model_type == 'reg':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
    elif model_type == 'class':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    else:
        print("Give a model type! model_type argument should be equal to 'reg' or 'class'")
        return None
    temp = dataframe[[X, y]].dropna()