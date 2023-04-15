import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
df.head()
df.info()
df.isna().sum()
df.nunique()
df.drop(df.columns[df.isnull().mean() > 0.7], axis=1, inplace=True)
df.shape
df.head()
df.drop(columns='PassengerId', inplace=True)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_trans = imp.fit_transform(df.select_dtypes('number'))
df_num = pd.DataFrame(imp_trans, columns=df.select_dtypes('number').columns)
df_num.head()
imp_ob = SimpleImputer(strategy='most_frequent')
imp_transob = imp_ob.fit_transform(df.select_dtypes('object'))
df_imp_object = pd.DataFrame(imp_transob, columns=df.select_dtypes('object').columns)
df_imp_object.head()
df = pd.concat([df_num, df_imp_object, df.select_dtypes('bool')], axis=1)
df.head()
df.isnull().sum()
corr = df.select_dtypes('number').corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr)
df['Transported'].corr(df['RoomService'])
df['Transported'].corr(df['Spa'])
df['Transported'].corr(df['VRDeck'])
df['Transported'].corr(df['FoodCourt'])
df['Transported'].corr(df['ShoppingMall'])
df.drop(columns=['VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt'], inplace=True)
df.shape
for i in df.select_dtypes('number').columns:
    plt.figure()
    plt.title(f'{i}')
    plt.boxplot(df[i], vert=False)
df = df[df['Age'] < 62]
df = df[df['RoomService'] < 2000]
df.head()
for i in df.select_dtypes('number').columns:
    plt.figure()
    plt.title(f'{i}')
    plt.boxplot(df[i], vert=False)
(majority_class, minority_class) = round(df['Transported'].value_counts(normalize=True), 2)
print(majority_class, minority_class)
target = 'Transported'
X = df.drop(columns=target)
y = df[target]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
acc_baseline = y_train.value_counts(normalize=True).max()
print('Baseline Accuracy:', round(acc_baseline, 2))
model = make_pipeline(OneHotEncoder(use_cat_names=True), LogisticRegression(max_iter=1000))