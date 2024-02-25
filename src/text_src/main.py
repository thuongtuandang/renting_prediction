
from create_datasets import load_dataset, create_tokenized_dataset
from utilities import model_evaluation, cross_validation, hyper_tune, histplot, plot
import pandas as pd
from lightgbm import LGBMRegressor
import pickle

# This will create train, validation and test set with text features
# Datasets are saved at /sample_data_text
# Make sure you run Notebooks/EDA_NoTextModels in advance for cleaned datasets
load_dataset(percentage=0.1)

# Create tokenized dataset
# WARNING: this takes more than 1 hour to run, even with 10% dataset
# vectorized datasets were saved there already, uncomment if you wish to run again
# create_tokenized_dataset()

# Load tokenized datasets
vect_df_train_sample = pd.read_csv('../../sample_data_text/vect_train_data_sample.csv')
vect_df_val_sample = pd.read_csv('../../sample_data_text/vect_val_data_sample.csv')

#  Define X_train, y_train, X_val, y_val
X_train = vect_df_train_sample.drop(columns=['totalRent']).values
y_train = vect_df_train_sample['totalRent'].values
X_val = vect_df_val_sample.drop(columns=['totalRent']).values
y_val = vect_df_val_sample['totalRent'].values
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

gbm = LGBMRegressor(num_leaves=31)
cross_validation(gbm, X_train, y_train)
gbm.fit(X_train, y_train)
model_evaluation(gbm, X_val, y_val)

param_grid = {
    'num_leaves': [25, 31 ,40],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [80, 100, 120]
}

best_gbm = hyper_tune(gbm, X_val, y_val, param_grid)

best_gbm.fit(X_train, y_train)

y_val_pred = model_evaluation(best_gbm, X_val, y_val)

# Plot the residuals
residuals = y_val - y_val_pred
histplot(residuals)
plot(residuals)

# Save the model to disk as a .pkl file
with open('../../models/best_gbm_text.pkl', 'wb') as file:
    pickle.dump(best_gbm, file)