import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
profile = ProfileReport(pd.read_csv('data/input/spaceship-titanic/train.csv'), title='Spaceship Titanic EDA', explorative=True)
profile.to_widgets()
profile.to_notebook_iframe()