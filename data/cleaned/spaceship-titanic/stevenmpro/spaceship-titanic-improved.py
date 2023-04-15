import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = 'data/input/spaceship-titanic/'
train = pd.read_csv(path + 'train.csv')
x_train = train.drop(columns='Transported')
y_train = train.loc[:, 'Transported']
x_test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')
train.head()
x_test.head()
print(f'train_set : {train.shape}')
print(f'test_set : {x_test.shape}')
y_train.head()
plt.figure()
plt.title('Ratio of the target')
plt.bar(x=['Transported', 'Non transported'], height=y_train.value_counts(), color=('turquoise', 'indianred'))

plt.figure()
sns.heatmap(train.corr(), annot=True)

color = ['lightcoral', 'bisque', 'olivedrab', 'paleturquoise', 'skyblue', 'plum']
train_num_col = train.select_dtypes('number')
height = [train_num_col[x].count() for x in train_num_col]
plt.bar(x=train_num_col.columns, height=height, color=color, align='center')
plt.xticks(rotation=45)
for (num, hei) in zip(range(train_num_col.shape[1]), height):
    plt.annotate(hei, (num - 0.2, hei + 50))
plt.title('Counting values in numerical features')

nb_rows = train_num_col.shape[1] // 3
plt.figure(figsize=(10, 10))
for (i, col) in enumerate(train_num_col, start=1):
    plt.subplot(nb_rows, 3, i)
    sns.boxplot(data=train, y=col, color=color[i - 1])
    plt.title(col)
    plt.yscale('log')

train.describe()
train.info()
np.random.seed(1234)
cmap = matplotlib.cm.get_cmap('Set3')
color = [cmap(np.random.rand()) for _ in range(train.shape[1])]
plt.bar(train.columns, train.nunique(), color=color)
plt.title('Nunique by col')
plt.xticks(rotation=90)
plt.yscale('log')

plt.bar(train.columns, train.isnull().sum(), color=color)
plt.title('Sum of NA by col')
plt.xticks(rotation=90)

categorical_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP']
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorial_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean', fill_value=0), StandardScaler())
preprocessor = make_column_transformer((categorial_pipeline, categorical_features), (numerical_pipeline, numerical_features))
y_train = LabelEncoder().fit_transform(y_train)
(train_set, val_train_set, test_set, val_test_set) = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
model = make_pipeline(preprocessor, LogisticRegression(solver='liblinear'))