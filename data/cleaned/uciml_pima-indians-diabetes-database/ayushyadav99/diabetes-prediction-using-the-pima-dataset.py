import pandas as pd
import numpy as np
import seaborn as sns

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('float_format', '{:f}'.format)
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.describe()
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df.isnull().sum()
for feature in features:
    number = np.random.normal(df[feature].mean(), df[feature].std() / 2)
    df[feature].fillna(value=number, inplace=True)
df.isnull().sum()
df.where(df < 0).count()
for feature in features:
    df.loc[df[feature] < 0, feature] = 0
df.where(df < 0).count()
df.loc[df['Insulin'] > 300].Insulin.count()
df.loc[df.Insulin > 300, 'Insulin'] = 300
df.loc[df['Insulin'] > 300].Insulin.count()
df.describe()
plot = scatter_matrix(df, alpha=0.2, figsize=(20, 20))
x = df.loc[:, features].values
y = df.loc[:, ['Outcome']].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=7)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', '  principal component 6', 'principal component 7'])
pca_df = pd.concat([principal_df, df[['Outcome']]], axis=1)
pca_df.describe()
plot = scatter_matrix(pca_df, alpha=0.2, figsize=(20, 20))

def PCA_split_dataset(pca_df):
    pca_df = pca_df.sample(frac=1)
    pca_X = pca_df[pca_df.columns[0:7]]
    pca_y = pca_df[pca_df.columns[7]]
    return train_test_split(pca_X, pca_y, test_size=0.2)
lr_accuracy = []
for i in range(500):
    (train_X, val_X, train_y, val_y) = PCA_split_dataset(pca_df)
    model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=0)