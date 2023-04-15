import pandas as pd
import seaborn as sns
data_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', na_values={'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 0})
data_df.head()
data_df.info()
data_df.describe().T
sns.pairplot(data_df, hue='Outcome')
correlation_matrix = data_df.corr()
sns.heatmap(correlation_matrix, cmap='Reds', annot=True)
isna_c = data_df.isna().sum()
isna_c[isna_c > 0]
data_df.groupby('Outcome').median().T
sns.displot(data_df.Glucose)
data_df.Glucose.fillna(data_df.groupby('Outcome')['Glucose'].transform('median'), inplace=True)
sns.displot(data_df.Glucose)
sns.pairplot(data_df[['Outcome', 'Glucose']], hue='Outcome')
data_df['bmi_cat'] = data_df.BMI.apply(lambda x: 'U' if x < 18.5 else 'N' if x < 25 else 'V' if x < 30 else 'O')
data_df['age_cat'] = data_df.Age.apply(lambda x: 'A' if x < 30 else 'B' if x < 40 else 'C' if x < 50 else 'D')
data_df.bmi_cat = data_df.bmi_cat.astype('category')
data_df.age_cat = data_df.age_cat.astype('category')
sns.pairplot(data_df[['bmi_cat', 'BloodPressure', 'age_cat']], hue='bmi_cat')
data_df[['bmi_cat', 'BloodPressure', 'age_cat']].groupby(['bmi_cat', 'age_cat']).median()
data_df.BloodPressure.fillna(data_df.groupby(['bmi_cat', 'age_cat'])['BloodPressure'].transform('median'), inplace=True)
sns.pairplot(data_df[['Outcome', 'BloodPressure']], hue='Outcome')
isna_c = data_df.isna().sum()
isna_c[isna_c > 0]
sns.boxplot(data=data_df[['bmi_cat', 'BloodPressure']], y='BloodPressure', x='bmi_cat')
import matplotlib.pyplot as plt
plt.scatter(data=data_df[['Age', 'BloodPressure']], y='BloodPressure', x='Age')
data_df.BMI.fillna(data_df.groupby(['Outcome', 'age_cat'])['BMI'].transform('median'), inplace=True)
sns.pairplot(data_df[['Outcome', 'BMI']], hue='Outcome')
isna_c = data_df.isna().sum()
isna_c[isna_c > 0]
data_df.SkinThickness.fillna(data_df.groupby(['Outcome', 'bmi_cat'])['SkinThickness'].transform('median'), inplace=True)
sns.pairplot(data_df[['Outcome', 'SkinThickness']], hue='Outcome')
sns.pairplot(data_df[['Outcome', 'Insulin']], hue='Outcome')
sns.boxplot(data=data_df[['Outcome', 'Glucose']], x='Outcome', y='Glucose')
data_df['glu_cat'] = data_df.Glucose.apply(lambda x: 'N' if x < 140 else 'U')
data_df.Insulin.fillna(data_df.groupby(['bmi_cat', 'glu_cat', 'Outcome'])['Insulin'].transform('median'), inplace=True)
sns.pairplot(data_df[['Outcome', 'Insulin']], hue='Outcome')
sns.pairplot(data_df, hue='Outcome')
data_df.info()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(data_df.drop(['Outcome', 'glu_cat', 'age_cat', 'bmi_cat'], axis=1), data_df.Outcome, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)