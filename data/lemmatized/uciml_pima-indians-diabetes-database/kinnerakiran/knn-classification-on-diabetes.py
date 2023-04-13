import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.sample(6)
data.info()
data.describe()
data.columns
colsToModify = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in colsToModify:
    print(col + ' - ')
    print(data[data[col] == 0][col].value_counts())
print('Count of zero entries in this column')
colsToModify = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
val = []
for col in colsToModify:
    val.append(len(data[data[col] == 0]))
zeroCount = pd.DataFrame(val, index=colsToModify, columns=['zeroCount'])
zeroCount
for col in colsToModify:
    data[col] = data[col].replace(0, np.NaN)
    mean = int(data[col].mean(skipna=True))
    data[col] = data[col].replace(np.NaN, mean)
data.head()
data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
pass
pass
pass
pass

def outlierDetectionbyZ_Score(data, col):
    (mean, std) = (data[col].mean(), data[col].std())
    high_threshold = mean + 3 * std
    low_threshold = mean - 3 * std
    print('High: ', high_threshold, 'Low: ', low_threshold)
    pass
    pass
    pass
    return [low_threshold, high_threshold]
threshold_insulin = outlierDetectionbyZ_Score(data, 'Insulin')

def getOutliersIndexbyZ_Score(data, col):
    (mean, std) = (data[col].mean(), data[col].std())
    high_threshold = mean + 3 * std
    low_threshold = mean - 3 * std
    thresholdLimit = [low_threshold, high_threshold]
    indices = np.where((data[col] < thresholdLimit[0]) | (data[col] > thresholdLimit[1]))[0]
    return indices
df = data.copy()
indices = getOutliersIndexbyZ_Score(df, 'Insulin')
df = df.drop(index=indices)
pass
pass
pass
df.info()
df.describe()
from sklearn.model_selection import train_test_split
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean')