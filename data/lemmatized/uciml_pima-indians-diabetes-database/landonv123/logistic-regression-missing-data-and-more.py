import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
diabetes_df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
diabetes_df.head(10)
diabetes_df.describe()
diabetes_df.isnull().sum()

def draw_histograms(dataframe, features, rows, cols):
    pass
    for (i, feature) in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='deepskyblue')
        ax.set_title(feature + ' Distribution', color='black')
    fig.tight_layout()
draw_histograms(diabetes_df, diabetes_df.columns, 4, 3)
diabetes_df.Outcome.value_counts()
sn.countplot(x='Outcome', data=diabetes_df)
sn.pairplot(data=diabetes_df)
pass
sn.heatmap(diabetes_df.corr(), annot=True, cmap='coolwarm', vmax=0.6)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = diabetes_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
X['Intercept'] = 1
vif = pd.DataFrame()
vif['variables'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif[0:-1])
from statsmodels.tools import add_constant as add_constant
diabetes_df_constant = add_constant(diabetes_df)
import scipy.stats as st
st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols = diabetes_df_constant.columns[:-1]
model = sm.Logit(diabetes_df.Outcome, diabetes_df_constant[cols])