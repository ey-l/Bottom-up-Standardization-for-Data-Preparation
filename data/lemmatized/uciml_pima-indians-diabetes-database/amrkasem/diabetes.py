import pandas as pd
import seaborn as sns
data_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv', na_values={'Glucose': 0, 'BloodPressure': 0, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 0})
data_df.head()
data_df.info()
data_df.describe().T
pass
correlation_matrix = data_df.corr()
pass
isna_c = data_df.isna().sum()
isna_c[isna_c > 0]
data_df.groupby('Outcome').median().T
pass
data_df.Glucose.fillna(data_df.groupby('Outcome')['Glucose'].transform('median'), inplace=True)
pass
pass
data_df['bmi_cat'] = data_df.BMI.apply(lambda x: 'U' if x < 18.5 else 'N' if x < 25 else 'V' if x < 30 else 'O')
data_df['age_cat'] = data_df.Age.apply(lambda x: 'A' if x < 30 else 'B' if x < 40 else 'C' if x < 50 else 'D')
data_df.bmi_cat = data_df.bmi_cat.astype('category')
data_df.age_cat = data_df.age_cat.astype('category')
pass
data_df[['bmi_cat', 'BloodPressure', 'age_cat']].groupby(['bmi_cat', 'age_cat']).median()
data_df.BloodPressure.fillna(data_df.groupby(['bmi_cat', 'age_cat'])['BloodPressure'].transform('median'), inplace=True)
pass
isna_c = data_df.isna().sum()
isna_c[isna_c > 0]
pass
import matplotlib.pyplot as plt
pass
data_df.BMI.fillna(data_df.groupby(['Outcome', 'age_cat'])['BMI'].transform('median'), inplace=True)
pass
isna_c = data_df.isna().sum()
isna_c[isna_c > 0]
data_df.SkinThickness.fillna(data_df.groupby(['Outcome', 'bmi_cat'])['SkinThickness'].transform('median'), inplace=True)
pass
pass
pass
data_df['glu_cat'] = data_df.Glucose.apply(lambda x: 'N' if x < 140 else 'U')
data_df.Insulin.fillna(data_df.groupby(['bmi_cat', 'glu_cat', 'Outcome'])['Insulin'].transform('median'), inplace=True)
pass
pass
data_df.info()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(data_df.drop(['Outcome', 'glu_cat', 'age_cat', 'bmi_cat'], axis=1), data_df.Outcome, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)