import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pass
pass
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.isnull().values.any()
df.head()
df.shape
df.columns
df.info()
df.describe()
pd.pivot_table(df, index=['Outcome'], aggfunc=[np.mean])
pd.pivot_table(df, index=['Outcome'], aggfunc=[np.std])
colnames = df.columns
pass
for i in range(4):
    x = colnames[i]
    ax[0, i].boxplot(df[str(x)], labels=[str(x)])
    pass
    ax[0, i].set_title(str(x) + '\nBoxplot')
for i in range(4, 8):
    x = colnames[i]
    ax[1, i - 4].boxplot(df[str(x)], labels=[str(x)])
    pass
    ax[1, i - 4].set_title(str(x) + '\nBoxplot')
pass
pass
for i in range(4):
    x = colnames[i]
    ax[0, i].hist(df[str(x)], color='red', bins=20)
    pass
    ax[0, i].set_title(str(x) + '\nHistogram')
for i in range(4, 8):
    x = colnames[i]
    ax[1, i - 4].hist(df[str(x)], color='red', bins=20)
    pass
    ax[1, i - 4].set_title(str(x) + '\nHistogram')
pass
pass
for i in range(4):
    x = colnames[i]
    ax[0, i].violinplot(df[str(x)])
    pass
    ax[0, i].set_title(str(x) + '\nViolinplot')
for i in range(4, 8):
    x = colnames[i]
    ax[1, i - 4].violinplot(df[str(x)])
    pass
    ax[1, i - 4].set_title(str(x) + '\nViolinplot')
pass
pass
for i in range(4):
    x = colnames[i]
    pass
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    pass
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
pass
pass
for i in range(4):
    x = colnames[i]
    pass
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    pass
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
pass
pass
for i in range(4):
    x = colnames[i]
    pass
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    pass
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
pass
pass
for i in range(4):
    x = colnames[i]
    pass
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    pass
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
pass
pass
for i in range(4):
    x = colnames[i]
    pass
    ax[0, i].set_title(str(x) + '\nDistribution by Outcome')
for i in range(4, 8):
    x = colnames[i]
    pass
    ax[1, i - 4].set_title(str(x) + '\nDistribution by Outcome')
pass
pass
cor = df.corr()
pass
pass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pass
pass
import warnings
warnings.filterwarnings('ignore')
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x = pd.DataFrame(x_scaled, columns=x.columns)
from sklearn.model_selection import train_test_split
(itrain_x, test_x, itrain_y, test_y) = train_test_split(x, y, random_state=56, stratify=y, test_size=0.1)
(train_x, valid_x, train_y, valid_y) = train_test_split(itrain_x, itrain_y, random_state=56, stratify=itrain_y, test_size=1 / 9)
print(train_x.shape[0] / x.shape[0], valid_x.shape[0] / x.shape[0], test_x.shape[0] / x.shape[0])
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
clf = KNN(n_neighbors=10)