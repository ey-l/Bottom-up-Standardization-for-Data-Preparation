import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
dados = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dados.head()
dados.shape
dados.columns
dados.dtypes
dados.describe(include='all').T.round(2)
dados.isna().sum()
dados.info()
dados.corr()
pass
mascara = np.triu(np.ones(dados.corr().shape)).astype(np.bool)
pass
dados.nunique()

def f_grafico(df, coluna, target):
    dados = df
    target = target
    coluna = coluna
    analise = dados[coluna]
    pass
    f.suptitle(pd.DataFrame(analise).columns[0])
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    print(pd.DataFrame(analise).describe().round(2).T)
f_grafico(dados, 'Pregnancies', 'Outcome')
f_grafico(dados, 'Glucose', 'Outcome')
f_grafico(dados, 'BloodPressure', 'Outcome')
f_grafico(dados, 'SkinThickness', 'Outcome')
f_grafico(dados, 'Insulin', 'Outcome')
f_grafico(dados, 'BMI', 'Outcome')
f_grafico(dados, 'DiabetesPedigreeFunction', 'Outcome')
f_grafico(dados, 'Age', 'Outcome')
dados['total'] = 1
dados.groupby('Outcome', as_index=False)['total'].count()
dados['total'] = 1
x = dados.groupby('Outcome', as_index=False)['total'].count()
x['Outcome'].tolist()
dados['total'] = 1
x = dados.groupby('Outcome', as_index=False)['total'].count()
labels = x['Outcome'].tolist()
sizes = x['total'].tolist()
pass
pass
pass
pass
pass
dados['total'] = 1
dados.groupby(by='Outcome')['total'].count().plot.bar()
pass
temp = dados.groupby('Outcome', as_index=False)['total'].count()
temp.rename(columns={'Outcome': 'Quantidade'}, inplace=True)
temp['Percentual'] = temp['total'] / dados.shape[0] * 100
temp
pass
pass
ax.set_facecolor('#fafafa')
pass
pass
pass
target = dados['Outcome']
target
explicativas = dados.drop('Outcome', axis=1)
explicativas
(x_treino, x_teste, y_treino, y_teste) = train_test_split(explicativas, target, test_size=0.3, random_state=42)
modelo_log = LogisticRegression(random_state=42, max_iter=400)