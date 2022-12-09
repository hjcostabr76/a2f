from sklearn.linear_model import LinearRegression

import joblib
import numpy as np
import pandas as pd

'''
==================================================================

    Build regressor model through .csv blendshape weights and meta human curves.
    - Regressor model is saved for outside usage;

==================================================================
'''

# path_root = './canto-01-long'
path_root = './canto-01-short'
path_mh_csv = os.path.join(path_root, '_weight_mh.csv')
path_blends_csv = os.path.join(path_root, '_weight_blends.csv')
path_model = os.path.join(path_root, 'regressors.joblib')



df_blends = pd.read_csv(path_blends_csv)
df_mh = pd.read_csv(path_mh_csv)

weights_blends = df_blends.values
weights_mh = df_mh.values

regressor = LinearRegression().fit(weights_blends, weights_mh)
joblib.dump(regressor, path_model) 