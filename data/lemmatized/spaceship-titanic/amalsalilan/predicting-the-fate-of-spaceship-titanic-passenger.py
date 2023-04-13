import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.info()
_input1.head()
df = _input1.copy()
df.describe(include='all')
df_processed = pd.DataFrame()
for column in df.columns:
    dtype = df[column].dtype
    if dtype == 'float64' or dtype == 'int64':
        df_processed[column] = df[column].fillna(df[column].median())
    else:
        df_processed[column] = df[column].fillna(df[column].mode()[0])
        df_processed[column] = pd.Categorical(df_processed[column]).codes
df_processed.describe()

def evaluate_dataset(df):
    """
  Evaluate the quality of a preprocessed dataset for a classification problem.
  
  Parameters:
  df (pandas.DataFrame): Preprocessed dataset with features and target column.
  
  Returns:
  float: Rating of the dataset out of 10.
  """
    target = df.columns[-1]
    class_counts = df[target].value_counts()
    class_balance = class_counts.max() / class_counts.sum()
    corr = df.drop(target, axis=1).corr()
    missing_values = df.isnull().sum().sum() / df.size
    score = 10 - (class_balance + missing_values + corr.abs().mean().mean())
    print(f'Data quality score: {score:.2f}/10')
    if class_balance < 0.8:
        print('The class balance is low. Consider oversampling or undersampling the minority class.')
    if missing_values > 0:
        print('There are missing values in the dataset. Consider imputing or dropping them.')
    if corr.abs().mean().mean() > 0.75:
        print('There is high correlation between some features. Consider removing correlated features or applying feature selection.')
evaluate_dataset(df_processed)
df_processed.describe(include='all')
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def balance_classes(df, target, method='oversampling'):
    """
  Balance the class distribution in a Pandas DataFrame.
  
  Parameters:
  df (pandas.DataFrame): DataFrame with features and target column.
  target (str): Name of the target column.
  method (str): Method to use for balancing the classes.
               Can be 'oversampling' (default) or 'undersampling'.
  
  Returns:
  pandas.DataFrame: DataFrame with balanced class distribution.
  """
    X = df.drop(target, axis=1)
    y = df[target]
    if method == 'oversampling':
        oversampler = SMOTE(random_state=0)
        (X_resampled, y_resampled) = oversampler.fit_resample(X, y)
    elif method == 'undersampling':
        undersampler = RandomUnderSampler(random_state=0)
        (X_resampled, y_resampled) = undersampler.fit_resample(X, y)
    else:
        raise ValueError(f'Invalid method: {method}')
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_resampled
df_balanced = balance_classes(df_processed, 'Transported')
df_balanced = balance_classes(df_processed, 'Transported', method='undersampling')
evaluate_dataset(df_balanced)
df_balanced.describe(include='all')
import matplotlib.pyplot as plt
plt.hist(df_balanced['Transported'])
plt.xlabel('Transported')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.hist(df_balanced['Age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
import seaborn as sns
sns.countplot(df_balanced['Transported'])
plt.xlabel('Transported')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.figure(figsize=(12, 8))
sns.heatmap(df_balanced.corr(), cmap='RdBu', annot=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
models = ['random_forest', 'gradient_boosting']
trained_models = {}
X = df_balanced.drop('Transported', axis=1)
y = df_balanced['Transported']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=0, stratify=y)
for model_type in models:
    if model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier()