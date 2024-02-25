# Here I will use the tokenized dataset from sample_data_text/vect_test_data_sample.csv
# Run src/text_src/main.py to create it again if you wish, but it would take an hour or more

import pandas as pd
import numpy as np
from joblib import load
from utilities import model_evaluation, histplot, plot

vect_df_test_sample = pd.read_csv('../../sample_data_text/vect_test_data_sample.csv')

X_test = vect_df_test_sample.drop(columns=['totalRent']).values
y_test = vect_df_test_sample['totalRent'].values

best_gbm_text = load('../../models/best_gbm_text.pkl')
y_test_pred = model_evaluation(best_gbm_text, X_test, y_test)
residuals = y_test - y_test_pred
histplot(residuals)
plot(residuals)