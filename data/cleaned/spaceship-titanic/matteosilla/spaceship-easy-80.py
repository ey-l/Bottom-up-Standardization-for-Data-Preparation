import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import category_encoders as ce
train_df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
PassengerId = test_df['PassengerId']
train_df.tail()
train_df.info()
print('_' * 40)
test_df.info()
train_df.describe()
sns.countplot(x='Transported', data=train_df, palette='PuBu')
plt.title('Distribution of Trasnported ')
plt.xlabel('Transported')
plt.ylabel('Count')

sns.displot(data=train_df, x='Age', hue='Transported', element='step', kde=True)
plt.title('Distribution of Age for Transported and Non-Transported people')
plt.xlabel('Age')
plt.ylabel('Density')

for df in [train_df, test_df]:
    df['Infants'] = df['Age'].apply(lambda x: 1 if x < 13 else 0)
sns.countplot(data=train_df, x='Infants', hue='Transported', palette='PuBu')
plt.title('Distribution of Age for Transported and Non-Transported people')
plt.xlabel('Infants')
plt.ylabel('Density')

(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
sns.violinplot(x='Transported', y='RoomService', data=train_df, ax=ax1, palette='PuBu')
ax1.set_title('Room Service for Transported and Non-Transported people', fontsize=15)
ax1.set_xlabel('Transported', fontsize=13)
ax1.set_ylabel('Room Service', fontsize=13)
ax1.set_xticklabels(['Non-Transported', 'Transported'], fontsize=13)
sns.boxplot(x='Transported', y='RoomService', data=train_df, ax=ax2, palette='PuBu')
ax2.set_xlabel('Transported', fontsize=13)
ax2.set_ylabel('RoomService', fontsize=13)
ax2.set_title('Transported vs RoomService Boxplot', fontsize=15)
ax2.set_xticklabels(['Non-Transported', 'Transported'], fontsize=13)

(fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
sns.violinplot(x='Transported', y='FoodCourt', data=train_df, ax=ax1, palette='PuBu', annot=True)
ax1.set_title('Food Court for Transported and Non-Transported people', fontsize=15)
ax1.set_xlabel('Transported', fontsize=13)
ax1.set_ylabel('Food Court', fontsize=13)
ax1.set_xticklabels(['Non-Transported', 'Transported'], fontsize=13)
sns.boxplot(x='Transported', y='FoodCourt', data=train_df, ax=ax2, palette='PuBu')
ax2.set_xlabel('Food Court', fontsize=13)
ax2.set_ylabel('RoomService', fontsize=13)
ax2.set_title('Transported vs FoodCourt Boxplot', fontsize=15)
ax2.set_xticklabels(['Non-Transported', 'Transported'], fontsize=13)

features = ['ShoppingMall', 'Spa', 'VRDeck']
(fig, axs) = plt.subplots(1, 3, figsize=(15, 5))
for (i, feature) in enumerate(features):
    sns.violinplot(x='Transported', y=feature, data=train_df, ax=axs[i], palette='PuBu')
    axs[i].set_title('Distribution of ' + feature)
    axs[i].set_xlabel('Transported')
    axs[i].set_ylabel(feature)

corr = train_df[['ShoppingMall', 'Spa', 'VRDeck', 'FoodCourt', 'RoomService']].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation of Features')
plt.xlabel('Features')
plt.ylabel('Features')

for df in [train_df, test_df]:
    df['LuxuryAmenities'] = df['ShoppingMall'] + df['FoodCourt'] + df['VRDeck'] + df['Spa'] + df['RoomService']
sns.displot(data=train_df, x='LuxuryAmenities', hue='Transported', element='step', kde=True, palette='PuBu')
plt.title('Distribution of Cabin Num for Transported and Non-Transported people')
plt.xlabel('Luxury Amenities')
plt.ylabel('Density')
plt.ylim(0, 400)
plt.xlim(0, 10000)

for df in [train_df, test_df]:
    df.loc[df['LuxuryAmenities'].notnull(), 'Buyer'] = df.loc[df['LuxuryAmenities'].notnull(), 'LuxuryAmenities'].map(lambda x: 1 if x > 0 else 0)
sns.countplot(x='HomePlanet', hue='Transported', data=train_df, palette='PuBu')
plt.title('Distribution of HomePlanet for Transported and Non-Transported people')
plt.xlabel('Transported')
plt.ylabel('Count')

sns.countplot(x='VIP', hue='Transported', data=train_df, palette='PuBu')
plt.title('Distribution of VIP for Transported and Non-Transported people')
plt.xlabel('VIP')
plt.ylabel('Count')

sns.countplot(x='CryoSleep', hue='Transported', data=train_df, palette='PuBu')
plt.title('Distribution of CryoSleep for Transported and Non-Transported people')
plt.xlabel('CryoSleep')
plt.ylabel('Count')

sns.countplot(x='Destination', hue='Transported', data=train_df, palette='PuBu')
plt.title('Distribution of Destination for Transported and Non-Transported people')
plt.xlabel('Destination')
plt.ylabel('Count')

for df in [train_df, test_df]:
    df['Cabin_Deck'] = df['Cabin'].str.split('/').str[0]
    df['Cabin_Num'] = df['Cabin'].str.split('/').str[1]
    df['Cabin_Side'] = df['Cabin'].str.split('/').str[2]
train_df.head()
sns.countplot(x='Cabin_Deck', hue='Transported', data=train_df, palette='PuBu')
plt.title('Distribution of Cabin Deck for Transported and Non-Transported people')
plt.xlabel('Cabin Deck')
plt.ylabel('Count')

sns.countplot(x='Cabin_Side', hue='Transported', data=train_df, palette='PuBu')
plt.title('Distribution of Cabin Side for Transported and Non-Transported people')
plt.xlabel('Cabin Side')
plt.ylabel('Count')

sns.distplot(train_df[train_df['Transported'] == True]['Cabin_Num'], label='Transported')
sns.distplot(train_df[train_df['Transported'] == False]['Cabin_Num'], label='Not Transported')
plt.legend()
plt.title('Distribution of Cabin Num for Transported and Non-Transported people')
plt.xlabel('Cabin Num')
plt.ylabel('Density')

for df in [train_df, test_df]:
    df['Group'] = df['PassengerId'].str.split('_').str[0]
train_df['GroupCount'] = train_df.groupby('Group')['Group'].transform('count')
test_df['GroupCount'] = test_df.groupby('Group')['Group'].transform('count')
sns.countplot(hue='Transported', x='GroupCount', data=train_df, palette='PuBu')
plt.title('Distribution of Passenger Group Count for Transported and Non-Transported people')
plt.xlabel('Group_Count')
plt.ylabel('Count')

for df in [train_df, test_df]:
    df['Alone'] = df['GroupCount'].map(lambda x: 1 if x == 1 else 0)
for df in [train_df, test_df]:
    nan_counts = df.isna().sum()
    nan_percentages = nan_counts / len(df) * 100
    print(nan_percentages)
    print('_' * 40)
import itertools
import math
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'Alone', 'Cabin_Deck', 'Cabin_Side']
comb = list(itertools.combinations(categorical_features, 2))
n = len(comb)
(fig, axes) = plt.subplots(math.ceil(n / 3), 3, figsize=(15, 10 * math.ceil(n / 3)))
axes = axes.flatten()
for (i, (col1, col2)) in enumerate(comb):
    gb = train_df.groupby([col1, col2])[col2].size().unstack().fillna(0)
    sns.heatmap(gb.T, annot=True, fmt='g', cmap='coolwarm', ax=axes[i])
    axes[i].set_title(f'{col1} vs {col2}')
for i in range(n, len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
grouped_df = train_df.groupby(['HomePlanet', 'Destination', 'Cabin_Deck'])['Cabin_Deck'].size().unstack().fillna(0)
plt.figure(figsize=(10, 4))
sns.heatmap(grouped_df, annot=True, fmt='g', cmap='coolwarm')
plt.title('Heatmap of Joint Distribution of HomePlanet, Destination, and Cabin Deck')

for df in [train_df, test_df]:
    mask = ((df['Destination'] == 'PSO J318.5-22') | df['Cabin_Deck'] == 'G') & df['HomePlanet'].isnull()
    df.loc[mask, 'HomePlanet'] = 'Earth'
    df.loc[df['HomePlanet'].isnull() & (df['Destination'] == 'TRAPPIST-1e'), 'HomePlanet'] = 'Mars'
    mask = ((df['Cabin_Deck'] == 'A') | (df['Cabin_Deck'] == 'B') | (df['Cabin_Deck'] == 'C') | (df['Cabin_Deck'] == 'T')) & df['HomePlanet'].isnull()
    df.loc[mask, 'HomePlanet'] = 'Europa'
    mask = (df['HomePlanet'] == 'Mars') & df['Destination'].isnull()
    df.loc[mask, 'Destination'] = 'TRAPPIST-1e'
    mask = (df['HomePlanet'] == 'Mars') & df['Cabin_Deck'].isnull()
    df.loc[mask, 'Cabin_Deck'] = 'F'
grouped_df = train_df.groupby(['CryoSleep', 'Buyer'])['CryoSleep'].size().unstack().fillna(0)
plt.figure(figsize=(10, 4))
sns.heatmap(grouped_df, annot=True, fmt='g', cmap='coolwarm')
plt.title('Heatmap of Joint Distribution of Cryosleep vs Buyer')

grouped_df = train_df.groupby(['Buyer', 'Infants'])['Infants'].size().unstack().fillna(0)
plt.figure(figsize=(10, 4))
sns.heatmap(grouped_df, annot=True, fmt='g', cmap='coolwarm')
plt.title('Heatmap of Joint Distribution of Buyer and Infants')

spendings_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for df in [train_df, test_df]:
    for feature in spendings_features:
        mask = ((df['CryoSleep'] == True) | (df['Age'] <= 12)) & df[feature].isnull()
        df.loc[mask, feature] = 0
        meanNoZero = df[df[feature] != 0][feature].mean()
        mask = (df['CryoSleep'] == False) & (df['Age'] > 12) & df[feature].isnull()
        df.loc[mask, feature] = meanNoZero
for df in [train_df, test_df]:
    df['LuxuryAmenities'] = df['ShoppingMall'] + df['FoodCourt'] + df['VRDeck'] + df['Spa'] + df['RoomService']
    df.loc[df['LuxuryAmenities'].notnull(), 'Buyer'] = df.loc[df['LuxuryAmenities'].notnull(), 'LuxuryAmenities'].map(lambda x: 1 if x > 0 else 0)
for df in [train_df, test_df]:
    mask = (df['LuxuryAmenities'] > 0) & df['CryoSleep'].isnull()
    df.loc[mask, 'CryoSleep'] = False
    mask = (df['LuxuryAmenities'] == 0) & df['CryoSleep'].isnull()
    df.loc[mask, 'CryoSleep'] = True
for df in [train_df, test_df]:
    nan_counts = df.isna().sum()
    nan_percentages = nan_counts / len(df) * 100
    print(nan_percentages)
    print('_' * 40)
categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
imputer = SimpleImputer(strategy='most_frequent')
for df in [train_df, test_df]:
    df_categorical_imputed = imputer.fit_transform(df[categorical_cols])
    df[categorical_cols] = df_categorical_imputed
imputer = KNNImputer()
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'LuxuryAmenities']
for df in [train_df, test_df]:
    df_imputed = imputer.fit_transform(df[numerical_features])
    df[numerical_features] = df_imputed
object_cols = train_df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in object_cols:
    for df in [train_df, test_df]:
        df[col] = le.fit_transform(df[col].astype(str))
train_df.info()
print('_' * 40)
test_df.info()
skewed_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'LuxuryAmenities']
for df in [train_df, test_df]:
    for feature in skewed_features:
        df[feature] = df[feature].map(lambda i: np.log(i) if i >= 1 else 0)
(fig, axs) = plt.subplots(2, 3, figsize=(15, 10))
for (i, feature) in enumerate(skewed_features):
    axs[i // 3][i % 3].hist(train_df[feature].dropna(), bins=30, color='blue', alpha=0.5, label='Transported')
    axs[i // 3][i % 3].hist(train_df[train_df['Transported'] == 0][feature].dropna(), bins=30, color='red', alpha=0.5, label='Not Transported')
    axs[i // 3][i % 3].set_title('Distribution of ' + feature)
    axs[i // 3][i % 3].set_xlabel(feature)
    axs[i // 3][i % 3].legend()

for df in [train_df, test_df]:
    df['GroupSpendings'] = df.groupby('Group')['LuxuryAmenities'].transform('sum')
sns.displot(data=train_df, x='GroupSpendings', hue='Transported', element='step')
plt.title('Distribution of total Group Spending for Transported and Non-Transported people')
plt.xlabel('Total group spending')
plt.ylabel('Density')
plt.xlim(0, 50)

for df in [train_df, test_df]:
    df['GroupSpendingsRate'] = df['GroupSpendings'] / df['GroupCount']
import scipy.stats as stats
numerical_features = train_df[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'LuxuryAmenities', 'Cabin_Num', 'GroupSpendings']]
target = train_df['Transported']
p_values = []
for feature in numerical_features:
    p = stats.ttest_ind(train_df[train_df['Transported'] == 1][feature], train_df[train_df['Transported'] == 0][feature]).pvalue
    p_values.append(p)
sns.barplot(x=numerical_features.columns, y=p_values)
plt.title('t-test of numerical features with target variable')
plt.xlabel('Numerical features')
plt.ylabel('p value')
plt.xticks(rotation=90)

cat_cols = train_df[['Transported', 'PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Cabin_Deck', 'Cabin_Side', 'Name', 'Infants', 'GroupCount']]
chi2_results = {}
for col in cat_cols:
    contingency_table = pd.crosstab(train_df['Transported'], train_df[col])
    (chi2, p, dof, expected) = stats.chi2_contingency(contingency_table)
    chi2_results[col] = [chi2, p]
chi2_results_df = pd.DataFrame(chi2_results, index=['chi2', 'p'])
sns.heatmap(chi2_results_df.T, annot=True, cmap='YlGnBu')
plt.title('Chi-Squared Test Results')

corr = train_df.corr().round(3)
sns.set(rc={'figure.figsize': (20, 8)})
sns.heatmap(corr[((corr >= 0.6) | (corr <= -0.6)) & (corr != 1)], annot=True, linewidths=0.5, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between features')
plt.xticks(rotation=90)

train_df.drop(columns=['PassengerId', 'Group', 'Name', 'Cabin', 'Cabin_Num', 'Buyer'], inplace=True)
test_df.drop(columns=['PassengerId', 'Group', 'Name', 'Cabin', 'Cabin_Num', 'Buyer'], inplace=True)
X_train = train_df.drop('Transported', axis=1)
Y_train = train_df['Transported'].astype(int)
X_test = test_df.copy()
(X_train.shape, Y_train.shape, X_test.shape)
logreg = LogisticRegression(max_iter=2500)