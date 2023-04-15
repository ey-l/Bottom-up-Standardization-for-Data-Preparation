import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_data.head()
train_data.describe().T
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
train_data.head().T
plt.figure(figsize=(30, 30))
sns.heatmap(train_data.corr(), annot=True, cmap='Pastel2_r')
train_data.isna().sum()
train_data.dtypes

def preprocess_data(df):
    for (label, content) in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label] = content.fillna(content.median())
        if not pd.api.types.is_numeric_dtype(content):
            df[label] = pd.Categorical(content).codes + 1
    return df
df = preprocess_data(df=train_data)
df.head()
df.isna().sum()
df.dtypes
df.columns
plt.scatter(df.YearBuilt, df.SalePrice, c='blue')
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')

plt.bar(df.YrSold, df.SalePrice, color='lightcoral')
plt.xlabel('YearSold')
plt.ylabel('SalePrice')

plt.scatter(df.GrLivArea, df.SalePrice, c='lightgreen', marker='*')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')

plt.scatter(df.LotArea, df.SalePrice, c='red', marker='^')
plt.xlabel('LotArea')
plt.ylabel('SalePrice')

plt.scatter(df.LotFrontage, df.SalePrice, c='lightblue', marker='+')
plt.xlabel('LotFrontage')
plt.ylabel('SalePrice')

labels = ('Average', 'Above Average', 'Good', 'Very Good', 'Below Average', 'Excellent', 'Fair', 'Very Excellent', 'Poor', 'Very Poor')
explode = (0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7)
NUM_COLORS = len(explode)
(fig1, ax1) = plt.subplots(figsize=(10, 10))
colors = ['lightskyblue', 'red', 'blue', 'green', 'gold', 'black', 'chocolate', 'brown', 'pink', 'maroon']
ax1.pie(df['OverallQual'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=30, colors=colors)
ax1.axis('equal')

fig = sns.barplot(x='OverallQual', y='SalePrice', data=df)
fig.set_xticklabels(labels=['Very Poor', 'Poor', 'Fair', 'Below Average', 'Average', 'Above Average', 'Good', 'Very Good', 'Excellent', 'Very Excellent'], rotation=90)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
from sklearn.model_selection import train_test_split, cross_val_score
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2, random_state=42)
(len(X_train), len(X_val), len(y_train), len(y_val))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
np.random.seed(42)
rg = RandomForestRegressor()