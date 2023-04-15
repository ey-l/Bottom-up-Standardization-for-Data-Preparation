import os
import numpy as np
import pandas as pd
targetName = 'item_cnt_month'
competitionDir = '/kaggle/input/competitive-data-science-predict-future-sales'
submission = pd.read_csv('_data/input/competitive-data-science-predict-future-sales/sample_submission.csv')
preds = []
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        if (dirname != competitionDir) & ('.csv' in filename):
            df = pd.read_csv(os.path.join(dirname, filename))
            if len(df) == len(submission):
                try:
                    preds.append(df[targetName])
                except Exception:
                    pass
submission[targetName] = np.array(preds).mean(axis=0).transpose()
