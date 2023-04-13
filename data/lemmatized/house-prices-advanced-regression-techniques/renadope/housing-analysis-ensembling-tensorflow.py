import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 115
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head()
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head()

def print_unique_values_df(df: pd.DataFrame):
    for col in list(df):
        print('Unique Values for {}:{}'.format(str(col), df[col].unique()))
        print('Num Unique Values for {}:{} values '.format(str(col), df[col].nunique()))
        print('dtype for {} is :{}'.format(str(col), df[col].dtypes))
        print('-' * 150)
print_unique_values_df(_input1)

def get_percentage_missing(df: pd.DataFrame, sort_val: bool=True, ascending: bool=False, col_name: str='percent_missing'):
    if sort_val == True:
        return pd.DataFrame(df.isnull().sum().sort_values(ascending=ascending) / len(df) * 100, columns=[col_name])
    elif sort_val == False:
        return pd.DataFrame(df.isnull().sum() / len(df) * 100, columns=[col_name])

def drop_cols_threshold_greater_than(df: pd.DataFrame, threshold: float=50, geq: bool=True):
    return df.drop(cols_with_nan_threshold_greater_than(df, threshold, geq=geq), axis=1)

def drop_cols_threshold_less_than(df: pd.DataFrame, threshold: float=50, leq: bool=True):
    return df.drop(cols_with_nan_threshold_less_than(df, threshold, leq=leq), axis=1)

def cols_with_nan_threshold_greater_than(df: pd.DataFrame, threshold: float=50, geq: bool=True):
    check_threshold(threshold)
    if geq:
        threshold_cols = [col for col in list(df.columns) if df[col].isnull().sum() / len(df) * 100 >= threshold]
    elif not geq:
        threshold_cols = [col for col in list(df.columns) if df[col].isnull().sum() / len(df) * 100 > threshold]
    return threshold_cols

def cols_with_nan_threshold_less_than(df: pd.DataFrame, threshold: float=50, leq: bool=True):
    check_threshold(threshold)
    if leq:
        threshold_cols = [col for col in list(df.columns) if df[col].isnull().sum() / len(df) * 100 <= threshold]
    elif not leq:
        threshold_cols = [col for col in list(df.columns) if df[col].isnull().sum() / len(df) * 100 < threshold]
    return threshold_cols

def check_threshold(threshold: float):
    if threshold < 0:
        raise ValueError('Threshold cannot be less than 0')
    if threshold > 100:
        raise ValueError('Threshold cannot be more than 100')

def show_percentage_missing(df: pd.DataFrame, sort_val: bool=True, ascending: bool=False, color: str='firebrick', col_name='percent_missing', title: str='Percentage of data missing.', opacity: float=1.0):
    if opacity < 0 or opacity > 1:
        raise ValueError('Invalid Opacity')
    df = get_percentage_missing(df, sort_val, ascending, col_name)
    fig = px.histogram(df, x=col_name, y=df.index, color_discrete_sequence=[color], opacity=opacity)
    fig.update_xaxes(title='Percentage (%) Missing')
    fig.update_yaxes(title='Column Name')
    fig.update_layout(title=title)
    fig.show()
print(f'Are the columns in the train and test the same?: {set(list(_input1.columns)) == set(list(_input0.columns))}')
train_set_cols = set(list(_input1.columns))
test_set_cols = set(list(_input0.columns))
print(f'Difference in columns: {train_set_cols.difference(test_set_cols)} ')
show_percentage_missing(df=_input1, ascending=False, sort_val=True, title='Percentage of data missing (Train)', color='purple')
show_percentage_missing(df=_input0, ascending=False, sort_val=True, title='Percentage of data missing (Test)', color='goldenrod')
df_train_numeric = _input1.select_dtypes(exclude='object')
df_test_numeric = _input0.select_dtypes(exclude='object')
dummy_col = np.random.random(len(df_test_numeric))
df_test_numeric = pd.concat([df_test_numeric, pd.Series(dummy_col).rename('SalePrice')], axis=1)
from sklearn.impute import SimpleImputer
old_lot_frontage = df_train_numeric['LotFrontage']
old_df_train_numeric = df_train_numeric
old_df_test_numeric = df_test_numeric
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(df_train_numeric)
df_train_numeric = pd.DataFrame(X_train, columns=list(df_train_numeric.columns))
X_test = imputer.transform(df_test_numeric)
df_test_numeric = pd.DataFrame(X_test, columns=list(df_test_numeric.columns))
df_test_numeric = df_test_numeric.drop(['SalePrice'], axis=1)
fig = make_subplots(rows=1, cols=2, subplot_titles=('Before Imputation', 'Simple Imputation (median)'))
fig.add_trace(go.Histogram(x=old_lot_frontage, showlegend=False, marker_color='#EB89B5'), row=1, col=1)
fig.add_trace(go.Histogram(x=df_train_numeric['LotFrontage'], showlegend=False, marker_color='#1E90ff', opacity=0.9), row=1, col=2)
fig.update_xaxes(title_text='Lot Frontage', row=1, col=1)
fig.update_xaxes(title_text='Lot Frontage', row=1, col=2)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Count', row=1, col=2)
fig.update_layout(title_text='Lot Frontage Distributions')
fig.show()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(old_df_train_numeric)
df_train_numeric = pd.DataFrame(X_train, columns=list(old_df_train_numeric.columns))
X_test = imputer.transform(old_df_test_numeric)
df_test_numeric = pd.DataFrame(X_test, columns=list(old_df_test_numeric.columns))
df_test_numeric = df_test_numeric.drop(['SalePrice'], axis=1)
fig = make_subplots(rows=1, cols=2, subplot_titles=('Before Imputation', 'KNN Imputation (neighbors = 5)'))
fig.add_trace(go.Histogram(x=old_lot_frontage, showlegend=False, marker_color='#EB89B5'), row=1, col=1)
fig.add_trace(go.Histogram(x=df_train_numeric['LotFrontage'], showlegend=False, marker_color='#1E90ff', opacity=0.9), row=1, col=2)
fig.update_xaxes(title_text='Lot Frontage', row=1, col=1)
fig.update_xaxes(title_text='Lot Frontage', row=1, col=2)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Count', row=1, col=2)
fig.update_layout(title_text='Lot Frontage Distributions')
fig.show()
df_train_categorical = _input1.select_dtypes(exclude=['float', 'int'])
df_test_categorical = _input0.select_dtypes(exclude=['float', 'int'])
threshold: float = 45
print(f"Categorical Columns With A Threshold of nan's greater than {threshold} % is {cols_with_nan_threshold_greater_than(df_train_categorical, threshold=threshold)} in Train set")
train_set_cols_drop = cols_with_nan_threshold_greater_than(df_train_categorical, threshold=threshold)
df_train_categorical = drop_cols_threshold_greater_than(df_train_categorical, threshold=threshold, geq=True)
df_train_categorical = df_train_categorical.apply(lambda x: x.fillna(x.value_counts().index[0]))
df_test_categorical = df_test_categorical.drop(train_set_cols_drop, axis=1)
df_test_categorical = df_test_categorical.apply(lambda x: x.fillna(x.value_counts().index[0]))
df_train_categorical = df_train_categorical[list(df_train_categorical.columns)].astype('category')
df_test_categorical = df_test_categorical[list(df_test_categorical.columns)].astype('category')
df_train_clean = pd.concat([df_train_numeric, df_train_categorical], axis=1)
df_test_clean = pd.concat([df_test_numeric, df_test_categorical], axis=1)
decade_bins = [1799]
decade_labels = []
while decade_bins[-1] < 2009:
    decade_labels.append(str(decade_bins[-1] + 1) + "'s")
    decade_bins.append(decade_bins[-1] + 10)
decade_bins.append(decade_bins[-1] + 1)
decade_labels.append(str(decade_bins[-1]) + "'s")
df_train_clean['decade'] = pd.cut(x=df_train_clean['YearBuilt'], bins=decade_bins, labels=decade_labels)
quarter_century_bins = [1799]
quarter_century_labels = []
while quarter_century_bins[-1] < 2000:
    quarter_century_labels.append(str(quarter_century_bins[-1] + 1) + "'s")
    quarter_century_bins.append(quarter_century_bins[-1] + 25)
df_train_clean['quarter_century'] = pd.cut(x=df_train_clean['YearBuilt'], bins=quarter_century_bins, labels=quarter_century_labels)
half_century_bins = [1799]
half_century_labels = []
while half_century_bins[-1] < 2000:
    half_century_labels.append(str(half_century_bins[-1] + 1) + "'s")
    half_century_bins.append(half_century_bins[-1] + 50)
df_train_clean['half_century'] = pd.cut(x=df_train_clean['YearBuilt'], bins=half_century_bins, labels=half_century_labels)
df_train_clean['century'] = pd.cut(x=df_train_clean['YearBuilt'], bins=[1799, 1899, 1999, 2010], labels=['19th Century', '20th Century', '21st Century'])
df_train_clean['qual_metric'] = pd.cut(x=df_train_clean['OverallQual'], bins=[0.0, 3.0, 5.0, 8.0, 10.0], labels=['Worst', 'Bad', 'Good', 'Excellent'])
df_train_clean['SalePriceBinned'] = pd.cut(x=df_train_clean['SalePrice'], bins=[0.0, 99999.99, 199999.99, 299999.99, 399999.99, 499999.99, 599999.99, 699999.99, 799999.99, 899999.99, 1000000.0], labels=['<100K', '100k - 199K', '200k - 299K', '300k - 399K', '400k - 499K', '500k - 599K', '600k - 699K', '700k - 799K', '800k - 899K', '900k - 1M'])
corr = df_train_clean.corr()
x = corr.nlargest(10 + 1, 'SalePrice').index
corr_df = df_train_clean[list(x)]
corr = corr_df.corr()
fig = px.imshow(corr)
fig.update_layout(title='Top 10 Features Correlated With Sale Price')
fig.show()
sale_price = df_train_clean['SalePrice']
iqr = np.quantile(sale_price, 0.75) - np.quantile(sale_price, 0.25)
lower = np.quantile(sale_price, 0.25) - 1.5 * iqr
upper = np.quantile(sale_price, 0.75) + 1.5 * iqr
df_train_no_outliers = df_train_clean[(df_train_clean['SalePrice'] > lower) & (df_train_clean['SalePrice'] < upper)]
print(f" Sale Price Skewness: {df_train_clean['SalePrice'].skew()}")
print(f" Sale Price Kurtosis: {df_train_clean['SalePrice'].kurtosis()}")
print(f" Sale Price (Outliers Removed) Skewness: {df_train_no_outliers['SalePrice'].skew()}")
print(f" Sale Price (Outliers Removed) Kurtosis: {df_train_no_outliers['SalePrice'].kurtosis()}")
from scipy.stats import shapiro
print(f" Sale Price (p-value)): {shapiro(df_train_clean['SalePrice']).pvalue}")
print(f" Sale Price (Outliers Removed) (p-value): {shapiro(df_train_no_outliers['SalePrice']).pvalue}")
print(f" Sale Price (Log Transformed)(p-value)): {shapiro(np.log(df_train_clean['SalePrice'])).pvalue}")
print(f" Sale Price (Log Transformed) (Outliers Removed) (p-value): {shapiro(np.log(df_train_no_outliers['SalePrice'])).pvalue}")
print(f" Sale Price (Square Root Transformed)(p-value)): {shapiro(np.power(df_train_clean['SalePrice'], 1 / 2)).pvalue}")
print(f" Sale Price (Square Root Transformed) (Outliers Removed) (p-value): {shapiro(np.power(df_train_no_outliers['SalePrice'], 1 / 2)).pvalue}")
print(f" Sale Price (Cube Root Transformed)(p-value)): {shapiro(np.power(df_train_clean['SalePrice'], 1 / 3)).pvalue}")
print(f" Sale Price (Cube Root Transformed) (Outliers Removed) (p-value): {shapiro(np.power(df_train_no_outliers['SalePrice'], 1 / 3)).pvalue}")
print(f" Sale Price (Cube Root Transformed)(p-value)): {shapiro(np.power(df_train_clean['SalePrice'], 1 / 4)).pvalue}")
print(f" Sale Price (Cube Root Transformed) (Outliers Removed) (p-value): {shapiro(np.power(df_train_no_outliers['SalePrice'], 1 / 4)).pvalue}")
fig = make_subplots(rows=4, cols=2, subplot_titles=('Original', 'With no Outliers', 'Original (Log Transformed)', 'With no Outliers (Log Transformed)', 'Original (sqrt Transformed)', 'With no Outliers (sqrt Transformed)', 'Original (cube rt Transformed)', 'With no Outliers (cube rt Transformed)'))
fig.add_trace(go.Histogram(x=df_train_clean['SalePrice'], showlegend=False, marker_color='#EB89B5'), row=1, col=1)
fig.add_trace(go.Histogram(x=df_train_no_outliers['SalePrice'], showlegend=False, marker_color='#1E90ff', opacity=0.9), row=1, col=2)
fig.add_trace(go.Histogram(x=np.log(df_train_clean['SalePrice']), showlegend=False, marker_color='#EB89B5'), row=2, col=1)
fig.add_trace(go.Histogram(x=np.log(df_train_no_outliers['SalePrice']), showlegend=False, marker_color='#1E90ff', opacity=0.9), row=2, col=2)
fig.add_trace(go.Histogram(x=np.power(df_train_clean['SalePrice'], 1 / 2), showlegend=False, marker_color='#EB89B5'), row=3, col=1)
fig.add_trace(go.Histogram(x=np.power(df_train_no_outliers['SalePrice'], 1 / 2), showlegend=False, marker_color='#1E90ff', opacity=0.9), row=3, col=2)
fig.add_trace(go.Histogram(x=np.power(df_train_clean['SalePrice'], 1 / 3), showlegend=False, marker_color='#EB89B5'), row=4, col=1)
fig.add_trace(go.Histogram(x=np.power(df_train_no_outliers['SalePrice'], 1 / 3), showlegend=False, marker_color='#1E90ff', opacity=0.9), row=4, col=2)
fig.update_xaxes(title_text='Sale Price', row=1, col=1)
fig.update_xaxes(title_text='Sale Price', row=1, col=2)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Count', row=1, col=2)
fig.update_layout(title_text='Sale Price Distributions', height=1200)
fig.show()
df_train_clean = df_train_clean.sort_values(by='OverallQual', ascending=True)
fig = px.ecdf(df_train_clean, x='SalePrice', color_discrete_sequence=['deeppink'])
fig.update_layout(title='ECDF Of Sale Price')
fig.add_vline(x=np.quantile(df_train_clean['SalePrice'], 0.1), line_width=3, line_dash='dash', line_color='blue', annotation_text=str(np.quantile(df_train_clean['SalePrice'], 0.1)) + 'K')
fig.add_vline(x=300000, line_width=3, line_dash='dash', line_color='blue', annotation_text='300K')
fig.add_vline(x=400000, line_width=3, line_dash='dash', line_color='blue', annotation_text='400K')
fig.add_vline(x=446261, line_width=3, line_dash='dash', line_color='blue', annotation_text='446K')
fig.add_vline(x=500000, line_width=3, line_dash='dash', line_color='blue', annotation_text='500K')
fig.add_vline(x=600000, line_width=3, line_dash='dash', line_color='blue', annotation_text='600K')
fig.add_vline(x=np.quantile(df_train_clean['SalePrice'], 0.999315), line_width=3, line_dash='dash', line_color='orange', annotation_text=str(np.floor(np.quantile(df_train_clean['SalePrice'], 0.999315))) + 'K')
fig.add_vline(x=np.median(df_train_clean['SalePrice']), line_dash='dot', line_color='green', line_width=5, annotation_text=str(np.floor(np.median(df_train_clean['SalePrice']))) + 'K (median)')
iplot(fig)
fig = px.ecdf(df_train_clean, x='SalePrice', color='qual_metric', color_discrete_sequence=['darkred', 'goldenrod', 'green', 'dodgerblue'])
fig.update_layout(title='ECDF Of Sale Price accoring to quality')
iplot(fig)
fig = px.bar(df_train_clean.groupby('decade')[['SalePrice']].count(), color_discrete_sequence=['dodgerblue'])
fig.update_layout(showlegend=False)
fig.update_xaxes(title='Decade Built')
fig.update_xaxes(title='Count')
fig.update_layout(title='Count of houses built per decade')
fig.show()
df_year_avg_price = pd.DataFrame(df_train_clean.groupby('YearBuilt')['SalePrice'].mean())
df_year_avg_price['pct_change'] = df_year_avg_price.pct_change() * 100
df_year_avg_price = df_year_avg_price.reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_year_avg_price['YearBuilt'], y=df_year_avg_price['pct_change'], name='spline', line_shape='spline', line=dict(color='#1E90ff')))
fig.add_hline(y=np.max(df_year_avg_price['pct_change']), line_dash='dot', row=3, col='all', annotation_text='Largest % Increase: ' + str(np.floor(np.max(df_year_avg_price['pct_change']))), annotation_position='bottom left')
fig.add_hline(y=np.min(df_year_avg_price['pct_change']), line_dash='dot', row=3, col='all', annotation_text='Largest % Decrease: ' + str(np.floor(np.min(df_year_avg_price['pct_change']))), annotation_position='bottom left')
fig.add_vline(x=1929, line_width=3, line_dash='dash', line_color='red', annotation_text='1929')
fig.add_vline(x=2008, line_width=3, line_dash='dash', line_color='red', annotation_text='2008')
fig.add_vrect(x0=1950, x1=1979, fillcolor='pink', opacity=0.3, annotation_text='Relatively Stable Period (1950-1979)', annotation_position='top left')
fig.update_traces(mode='lines+markers')
fig.update_xaxes(title='Year Built')
fig.update_yaxes(title='Percentage Change')
fig.update_layout(title='Average Sale Price Percentage Change per year')
fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))
df_train_clean = df_train_clean.sort_values(by='YearBuilt', ascending=True)
fig = make_subplots(rows=2, cols=2, subplot_titles=('Per Century', 'Per Quarter Century', 'Per Half Century', 'Per Decade'))
fig.add_trace(go.Box(x=df_train_clean['century'], y=df_train_clean['SalePrice'], showlegend=False, boxpoints='all'), row=1, col=1)
fig.add_trace(go.Box(x=df_train_clean['quarter_century'], y=df_train_clean['SalePrice'], showlegend=False, boxpoints='all'), row=1, col=2)
fig.add_trace(go.Box(x=df_train_clean['half_century'], y=df_train_clean['SalePrice'], showlegend=False, boxpoints='all'), row=2, col=1)
fig.add_trace(go.Box(x=df_train_clean['decade'], y=df_train_clean['SalePrice'], showlegend=False, boxpoints='all'), row=2, col=2)
fig.update_xaxes(title_text='Century', row=1, col=1)
fig.update_xaxes(title_text='Quarter Century', row=1, col=2)
fig.update_xaxes(title_text='Half Century', row=2, col=1)
fig.update_xaxes(title_text='Decade', row=2, col=2)
fig.update_yaxes(title_text='Sale Price', row=1, col=1)
fig.update_yaxes(title_text='Sale Price', row=1, col=2)
fig.update_yaxes(title_text='Sale Price', row=2, col=1)
fig.update_yaxes(title_text='Sale Price', row=2, col=2)
fig.update_layout(title_text='House Sale Price Distributions per timeframe', height=700)
fig.show()
df_train_clean = df_train_clean.sort_values(by='OverallQual', ascending=True)
fig = px.density_contour(df_train_clean, x='GrLivArea', y='SalePrice', color='qual_metric', color_discrete_sequence=['darkred', 'goldenrod', 'green', 'dodgerblue'], labels={'qual_metric': 'Quality'})
fig.update_traces(contours_showlabels=True)
fig.update_layout(title='Sale Price vs Ground Living Area')
fig.show()
fig = px.scatter(df_train_clean, x='GrLivArea', y='SalePrice', facet_col='qual_metric', color='TotRmsAbvGrd', opacity=0.99, color_continuous_scale='teal', trendline='ols')
fig.update_layout(title='Sale Price vs Ground Living Area')
fig.show()
fig = px.violin(df_train_clean, y='GrLivArea', color='qual_metric', color_discrete_sequence=['darkred', 'goldenrod', 'green', 'dodgerblue'])
fig.update_layout(title='Distribution of Ground Living Area per House Quality')
fig.show()
fig = px.box(df_train_clean, y='TotRmsAbvGrd', color='qual_metric', points='outliers', color_discrete_sequence=['darkred', 'goldenrod', 'green', 'dodgerblue'])
fig.update_layout(title='Distribution of Total Rooms Above Ground per House Quality')
fig.show()
df_train_clean = df_train_clean.sort_values(by='OverallQual', ascending=True)
fig = px.scatter(df_train_clean, x='1stFlrSF', y='SalePrice', facet_col='qual_metric', opacity=0.5, color='FullBath', color_continuous_scale='deep')
fig.update_layout(title='Sale Price vs 1st Floor Square Footage')
fig.show()
fig = px.violin(df_train_clean, y='1stFlrSF', color='qual_metric', color_discrete_sequence=['darkred', 'goldenrod', 'green', 'dodgerblue'])
fig.update_layout(title='1st Floor Square Footage Distribution per House Quality')
fig.show()
fig = px.imshow(pd.crosstab(df_train_clean['SalePriceBinned'], df_train_clean['Neighborhood']), color_continuous_scale='rdbu')
fig.update_layout(title='Price Density per Neighborhood')
fig.show()
fig = px.imshow(pd.crosstab(df_train_clean['SalePriceBinned'], df_train_clean['BldgType']), color_continuous_scale='teal')
fig.update_layout(title='Price Density per Building Type')
fig.show()
fig = px.imshow(pd.crosstab(df_train_clean['SalePriceBinned'], df_train_clean['decade']), color_continuous_scale='picnic')
fig.update_layout(title='Price Density per Decade')
fig.show()
df_train_clean = df_train_clean.sort_values(by='OverallQual', ascending=True)
fig = px.parallel_categories(df_train_clean, dimensions=['Neighborhood', 'SalePriceBinned'], color='YearBuilt', color_continuous_scale=px.colors.sequential.Plotly3)
fig.show()
fig = px.parallel_coordinates(df_train_clean, dimensions=['GarageArea', 'GarageCars', 'GrLivArea', 'TotRmsAbvGrd', 'KitchenAbvGr', 'FullBath', 'SalePrice'], labels={'GrLivArea': 'Living Area', 'GarageCars': 'Garage Capacity', 'KitchenAbvGr': "# Kit'en abv Grd Flr", 'FullBath': '# full baths', 'TotRmsAbvGrd': '# Rooms Above Ground', 'GarageArea': 'Garage Area', 'SalePrice': 'Sale Price'}, color_continuous_scale='agsunset', color='LotArea')
fig.show()
fig = px.parallel_categories(df_train_clean, dimensions=['Neighborhood', 'Street', 'BldgType', 'FullBath', 'qual_metric', 'SalePriceBinned'], color='YearBuilt', color_continuous_scale=px.colors.sequential.Jet)
fig.update_layout(title='Neighborhood -> Street -> Bldg Type -> # Full Baths -> Quality Rating -> Sale Price')
fig.show()
X_train_temp = df_train_clean.drop(['Id', 'SalePrice', 'qual_metric', 'SalePriceBinned', 'decade', 'quarter_century', 'half_century', 'century'], axis=1)
y_train = df_train_clean['SalePrice']
test_id_cols = list(df_test_clean['Id'])
X_test_temp = df_test_clean.drop(['Id'], axis=1)
temp_concat = pd.concat([X_train_temp, X_test_temp], axis=0)
temp_concat = pd.get_dummies(temp_concat)
X_train = temp_concat.iloc[0:1460, :]
X_test = temp_concat.iloc[1460:, :]
print(X_train.shape[1] == X_test.shape[1])
print(X_train.shape)
print(X_test.shape)
print(set(X_train.columns) - set(X_test.columns))

def scaled_df(X_scaled, y_target, list_of_cols):
    scaled_df = pd.DataFrame(X_scaled, columns=list_of_cols)
    scaled_df = pd.concat([scaled_df, y_target], axis=1)
    return scaled_df

def scale_comparision(scaled_df, orig_df, x_var, y_var, x_title, y_title, scaler_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Without Scaling', 'With ' + str(scaler_name)))
    fig.add_trace(go.Scattergl(x=orig_df[x_var], y=orig_df[y_var], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scattergl(x=scaled_df[x_var], y=scaled_df[y_var], mode='markers'), row=1, col=2)
    fig.update_layout(title_text='Comparision of Unscaled and Scaled Data ')
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    fig.update_layout(showlegend=False)
    fig.show()
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train_std_scaled = std_scaler.fit_transform(X_train)
X_train_std_scaled.shape
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X_train_min_max_scaled = min_max_scaler.fit_transform(X_train)
X_train_min_max_scaled.shape
from sklearn.preprocessing import MaxAbsScaler
max_abs_scaler = MaxAbsScaler()
X_train_max_abs_scaled = max_abs_scaler.fit_transform(X_train)
X_train_max_abs_scaled.shape
scale_comparision(scaled_df(X_train_max_abs_scaled, y_train, list(X_train.columns)), df_train_clean, 'LotArea', 'GrLivArea', 'Lot Area', 'Ground Living Area', 'MaxAbsScaler')
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X_train_robust_scaled = robust_scaler.fit_transform(X_train)
X_train_robust_scaled.shape
scale_comparision(scaled_df(X_train_robust_scaled, y_train, list(X_train.columns)), df_train_clean, 'LotArea', 'GrLivArea', 'Lot Area', 'Ground Living Area', 'RobustScaler')
from sklearn.preprocessing import PowerTransformer
pow_transformer = PowerTransformer(method='yeo-johnson')
X_train_pow_transform = pow_transformer.fit_transform(X_train)
X_train_pow_transform.shape
from sklearn.preprocessing import QuantileTransformer
qt_transformer = QuantileTransformer(output_distribution='uniform', random_state=RANDOM_STATE)
X_qt_transform_uni = qt_transformer.fit_transform(X_train)
qt_transformer = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
X_qt_transform_normal = qt_transformer.fit_transform(X_train)
X_qt_transform_normal.shape
scale_comparision(scaled_df(X_qt_transform_normal, y_train, list(X_train.columns)), df_train_clean, 'LotArea', 'GrLivArea', 'Lot Area', 'Ground Living Area', 'QuantileTransformer(Normal)')
from sklearn.preprocessing import Normalizer
normalizer_l1 = Normalizer(norm='l1')
X_train_normalize_l1 = normalizer_l1.fit_transform(X_train)
X_train_normalize_l1.shape
normalizer_l2 = Normalizer(norm='l2')
X_train_normalize_l2 = normalizer_l2.fit_transform(X_train)
X_train_normalize_l2.shape
scale_comparision(scaled_df(X_train_normalize_l2, y_train, list(X_train.columns)), df_train_clean, 'LotArea', 'GrLivArea', 'Lot Area', 'Ground Living Area', 'l2 Normalization')
normalizer_max = Normalizer(norm='max')
X_train_normalize_max = normalizer_max.fit_transform(X_train)
X_train_normalize_max.shape
from sklearn.decomposition import PCA

def disp_explained_variance_ratio(X, title):
    pca = PCA(n_components=0.9999)