import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
pass
"import os\n\nfor dirname, _, filenames in os.walk('/kaggle/input'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename)) "
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df = df.rename(columns={'BloodPressure': 'BP', 'DiabetesPedigreeFunction': 'DPF'})
df.info()
df.describe()
zeroCols = ['Glucose', 'BP', 'SkinThickness', 'Insulin', 'BMI']
df2 = df.copy()
df2[zeroCols] = df2[zeroCols].replace(0, np.NaN)
df2.head()
outcomes = df2['Outcome'].value_counts()
print(outcomes)
df.describe()
df2.describe()
null_values = df2.isna().sum() / len(df2) * 100
null_values.drop(labels=['Pregnancies', 'DPF', 'Age', 'Outcome'], inplace=True)
print('Column Name' + '     ' + '% of Null Values\n')
print(null_values)
dp = df2.groupby('Outcome').count()
outcome_0 = dp.loc[0, :]
outcome_1 = dp.loc[1, :]
print('Column Name' + '     ' + 'Outcome 1 to Outcome 2 Data Points Ratio\n')
print(outcome_1 / outcome_0)
hist = df2.hist(figsize=(20, 20))
df2['Glucose'].fillna(df2['Glucose'].median(), inplace=True)
df2['BMI'].fillna(df2['BMI'].median(), inplace=True)
df2['BP'].fillna(df2['BP'].mean(), inplace=True)
df2['Insulin'].fillna(df2['Insulin'].median(), inplace=True)
df2['SkinThickness'].fillna(df2['SkinThickness'].mean(), inplace=True)
histZR = df2.hist(figsize=(20, 20))
pass
df2.corr()
pass
out0 = df2[df2['Outcome'] == 0]
out1 = df2[df2['Outcome'] == 1]
out0.describe()
out1.describe()
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
features = df2.iloc[:, :8].values
target = df2.loc[:, 'Outcome'].values
testSize = 0.3
trainSize = 0.7
validSize = 0.5
rs = 42
(x_train, x, y_train, y) = train_test_split(features, target, train_size=trainSize, random_state=rs)
(x_val, x_test, y_val, y_test) = train_test_split(x, y, train_size=validSize, random_state=rs)
print(f'# of Training Data:{len(x_train)}\n# of Validation Data: {len(x_val)}\n# of Test Data:{len(x_test)}')
scaler = RobustScaler()
xTrain_scaled = scaler.fit_transform(x_train)
xVal_scaled = scaler.fit_transform(x_val)
xTest_scaled = scaler.fit_transform(x_test)

def modeleval(yTrue, yPredict, print_metrics, modelname):
    auc = roc_auc_score(yTrue, yPredict)
    cm = confusion_matrix(yTrue, yPredict)
    (tn, fp, fn, tp) = confusion_matrix(yTrue, yPredict).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    acc = accuracy_score(yTrue, yPredict)
    mm = {'AUC': auc, 'Confusion Matrix': cm, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'TPR': tpr, 'TNR': tnr, 'Accuracy': acc}
    if print_metrics:
        print(f"Sensitivity:{mm['TPR']}\n\nSpecificity:{mm['TNR']}\n\nAUC of ROC:{mm['AUC']}\n\nAccuracy:{mm['Accuracy']}\n\n")
        x = pd.crosstab(yTrue, yPredict, rownames=['True'], colnames=['Predicted'], margins=True)
        print(f'{x}\n')
        plot_confusion_matrix(yTrue, yPredict, classes=np.array(['No Diabetes', 'Diabetes']), title='Confusion matrix: ' + modelname)
    return mm

def plotroc(yvt, yvp, modelname):
    (f, t, thresh) = roc_curve(yvt, yvp)
    roc_auc = auc(f, t)
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass

pass
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    'if normalize:\n        cm = cm.astype(\'float\') / cm.sum(axis=1)[:, np.newaxis]\n        print("Normalized confusion matrix")\n    else:\n        print(\'Confusion matrix, without normalization\')\n\n    print(cm) '
    pass
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    pass
    pass
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    return ax
svm_ = SVC(kernel='rbf', class_weight='balanced', random_state=1)