import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.info(verbose=True)
df.describe()
sns.set_style('whitegrid')
box_plot = sns.boxplot(x='Outcome', y='Insulin', data=df)
medians = df.groupby(['Outcome'])['Insulin'].median()
vertical_offset = df['Insulin'].median() * 0.05
for xtick in box_plot.get_xticks():
    box_plot.text(xtick, medians[xtick] + vertical_offset, medians[xtick], horizontalalignment='center', size='x-small', color='w', weight='semibold')
print('Since 0 values appear in both outcomes ,its not an indicator of the insulin level and should be considered null value')
zero_attributes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def zero_values(df, zero_attributes):
    for i in zero_attributes:
        df_count = df.loc[df[i] == 0]
        x = df_count[i].count()
        print(f'The Number of zero values in column {i} is {x}')
zero_values(df, zero_attributes)
fig = plt.figure(figsize=(20, 12))
ax = fig.gca()
df.hist(ax=ax)

print('we can see that most of the columns are skewed')
di = {0.0: 0, 1.0: 1}
sns.pairplot(df.replace({'Outcome': di}), hue='Outcome')

c = [0, 1, 2, 3]
r = [0, 1]
cols_index = 0
cols = df.columns[:-1]
(fig, axs) = plt.subplots(2, 4, figsize=(20, 12))
for i in r:
    for j in c:
        box_plot = sns.boxplot(x='Outcome', y=cols[cols_index], data=df, ax=axs[i, j])
        medians = df.groupby(['Outcome'])[cols[cols_index]].median()
        vertical_offset = df[cols[cols_index]].median() * 0.05
        cols_index += 1
df_copy = df.copy()
X = df_copy.iloc[:, :-1]
y = df_copy.iloc[:, -1]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
missing_values = ['Glucose', 'BloodPressure', 'BMI']
drop_columns = ['Insulin', 'SkinThickness']

def preprocessing(df, missing_values, drop_columns):
    df.drop(columns=drop_columns, inplace=True)
    for col in missing_values:
        val = df[col].mean()
        df[col] = df[col].replace(0, val)
    return df
X_train_processed = preprocessing(X_train, missing_values, drop_columns)
X_test_processed = preprocessing(X_test, missing_values, drop_columns)
scaler = StandardScaler()