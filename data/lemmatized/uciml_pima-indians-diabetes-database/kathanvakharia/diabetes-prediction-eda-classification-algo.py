import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head()
dataset.shape
pass
pass
dataset.isna().any()
dataset.info()
dataset.describe()
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_not_accepted:
    dataset[col].replace(0, np.nan, inplace=True)
dataset.head(n=10)
pass
pass
for p in ax.patches:
    ax.annotate(text=f'{p.get_width():.0f}', xy=(p.get_width(), p.get_y() + p.get_height() / 2), xytext=(5, 0), textcoords='offset points', ha='left', va='center')
pass
for col in zero_not_accepted:
    dataset[col].replace(np.nan, dataset[col].mean(), inplace=True)
dataset.describe()
pass
pass
dataset['Outcome'].value_counts()
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()
print(X)
print(y)
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=0)
(X_r, y_r) = smk.fit_resample(X, y)
from collections import Counter
print(f'Initial counts: {Counter(y)}')
print(f'Resampled Counts: {Counter(y_r)}')
print(X_r.shape, y_r.shape)
X = X_r
y = y_r
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_train)
print(y_train.shape)
print(y_train)
print(X_test.shape)
print(X_test)
print(y_test.shape)
print(y_test)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
pd.DataFrame(X_train, columns=dataset.columns[:-1]).describe()
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def disp_cm(y_test, y_pred) -> float:
    """Displays the confusion matrix in the form of heatmap.
    
    Parameters:
    y_test (array-like): list of true labels
    y_pred (array-like): list of predicted labels
    
    Returns:
    acc_score (float): Accuracy score 
    """
    acc_score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    pass
    pass
    pass
    pass
    return acc_score

def judge_clf(classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test) -> float:
    """Fits the `classifier` to `X_train`, `y_train` and generate an elegant 
    classification report using `X_test` and `y_test`.
    
    Parameters:
    classifer : classifier obj implementing 'fit' method.
    X_train (array-like): 2D-array of input features of Training Set.
    y_train (array-like): list of target features of Training Set.
    X_test  (array-like): 2D-array of input features of Testing Set.
    y_test  (array-like): list of target features of Testing Set.
    
    Returns:
    acc_score (float): Accuracy score 
    """