import pandas as pd

from dataprep.eda import create_report
df = pd.read_csv('data/input/spaceship-titanic/test.csv')
print(df.shape)
report = create_report(df, title='Spaceship Titanic Training Data')
report.save('../Spaceship_Titanic_EDA.html')
report.show_browser()
report.show()