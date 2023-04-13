import datetime as dt
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from dataprep.eda import plot, plot_correlation, create_report, plot_missing
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
plot(_input1)
create_report(_input1)
plot(_input1, 'Age')
plot(_input1, 'Age', 'Transported')
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
dftc = AV.AutoViz(filename='', sep='', depVar='Transported', dfte=_input1, header=0, verbose=1, lowess=False, chart_format='png', max_rows_analyzed=300000, max_cols_analyzed=30)
from pandas_profiling import ProfileReport
ProfileReport(_input1)