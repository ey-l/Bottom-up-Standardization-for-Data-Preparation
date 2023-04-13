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
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input1.head()
_input1.info()
_input1.isna().sum()
_input1.nunique()
_input1 = _input1.drop(_input1.columns[_input1.isnull().mean() > 0.7], axis=1, inplace=False)
_input1.shape
_input1.head()
_input1 = _input1.drop(columns='PassengerId', inplace=False)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_trans = imp.fit_transform(_input1.select_dtypes('number'))
df_num = pd.DataFrame(imp_trans, columns=_input1.select_dtypes('number').columns)
df_num.head()
imp_ob = SimpleImputer(strategy='most_frequent')
imp_transob = imp_ob.fit_transform(_input1.select_dtypes('object'))
df_imp_object = pd.DataFrame(imp_transob, columns=_input1.select_dtypes('object').columns)
df_imp_object.head()
_input1 = pd.concat([df_num, df_imp_object, _input1.select_dtypes('bool')], axis=1)
_input1.head()
_input1.isnull().sum()
corr = _input1.select_dtypes('number').corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr)
_input1['Transported'].corr(_input1['RoomService'])
_input1['Transported'].corr(_input1['Spa'])
_input1['Transported'].corr(_input1['VRDeck'])
_input1['Transported'].corr(_input1['FoodCourt'])
_input1['Transported'].corr(_input1['ShoppingMall'])
_input1 = _input1.drop(columns=['VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt'], inplace=False)
_input1.shape
for i in _input1.select_dtypes('number').columns:
    plt.figure()
    plt.title(f'{i}')
    plt.boxplot(_input1[i], vert=False)
_input1 = _input1[_input1['Age'] < 62]
_input1 = _input1[_input1['RoomService'] < 2000]
_input1.head()
for i in _input1.select_dtypes('number').columns:
    plt.figure()
    plt.title(f'{i}')
    plt.boxplot(_input1[i], vert=False)
(majority_class, minority_class) = round(_input1['Transported'].value_counts(normalize=True), 2)
print(majority_class, minority_class)
target = 'Transported'
X = _input1.drop(columns=target)
y = _input1[target]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
acc_baseline = y_train.value_counts(normalize=True).max()
print('Baseline Accuracy:', round(acc_baseline, 2))
model = make_pipeline(OneHotEncoder(use_cat_names=True), LogisticRegression(max_iter=1000))