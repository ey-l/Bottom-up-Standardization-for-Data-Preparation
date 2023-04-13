import pandas as pd
from dataprep.eda import create_report
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(_input0.shape)
report = create_report(_input0, title='Spaceship Titanic Training Data')
report.save('../Spaceship_Titanic_EDA.html')
report.show_browser()
report.show()