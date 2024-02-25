# Here you have two options: 
# 1. Run main.py to create test set and proceed with filling methods, scaler, and label encoding
# 2. df_test.csv in data/ is already cleaned and you can proceed use it directly

import pandas as pd
from process_train_data import remove_columns, median_fill, fill_totalRent, remove_baseRent, fill_text_features, label_encoding, train_imputer, train_scaler
from joblib import load
import pickle
from process_test_data import fill_with_trained_imputer, label_encoding_test
from utilities import model_evaluation, histplot, plot

# Option 1: the input is your test data
# I assume X_test do not contain text features 
# Make sure to exclude the target variable totalRent and baseRent
# Make sure the label y is not NaN 
# totalRent shoud be meaningful (totalRent <= 20000) for better predictions

# Edit according to your path
X_test = pd.read_csv('../../test_data/X_test.csv')

# Columns to drop
# Column 1 list have very high mising values
columns_1_drop = ['telekomHybridUploadSpeed', 'noParkSpaces', 'petsAllowed', 
           'thermalChar', 'numberOfFloors', 'heatingCosts', 'energyEfficiencyClass',
          'lastRefurbish', 'electricityBasePrice', 'electricityKwhPrice']

# Column 2 list are duplicated columns
columns_2_drop = ['street','streetPlain', 'geo_bln', 'scoutId', 'date', 'geo_krs', 'houseNumber',
          'yearConstructedRange', 'baseRentRange', 'noRoomsRange', 'livingSpaceRange',
          'telekomTvOffer', 'firingTypes']

# Numerical features
num_features = ['serviceCharge', 'picturecount', 'pricetrend', 'telekomUploadSpeed', 
                'yearConstructed', 'livingSpace', 'noRooms', 'floor']

# Text features
text_features = ['description', 'facilities']

# Those columns are to fill with medians
num_missing_cols = ['serviceCharge', 'yearConstructed', 'pricetrend', 'telekomUploadSpeed', 'floor']

cat_features = ['regio1', 'heatingType', 'newlyConst', 'balcony', 'hasKitchen', 'cellar', 'condition', 
                'interiorQual', 'lift', 'typeOfFlat', 'garden', 'regio2', 'regio3', 'geo_plz']

# Those columns are to fill with random forest
cat_missing_columns = ['heatingType', 'condition', 'typeOfFlat', 'interiorQual']

# Preprocess
# Drop columns
remove_columns(X_test, columns_1_drop)
remove_columns(X_test, columns_2_drop)
# remove_columns(X_test, text_features)

# Fill with medians
medians_by_regio2 = load('../../imputers/medians_by_regio2.pkl')
for col in num_missing_cols:
    # For each row in df_val, if the value in the target column is missing,
    # fill it with the median from medians_by_regio2 where 'regio2' matches.
    X_test[col] = X_test.apply(
        lambda row: medians_by_regio2.loc[medians_by_regio2['regio2'] == row['regio2'], col].values[0]
        if pd.isnull(row[col]) else row[col], axis=1)

print("Finish median fillings")

# Label encoder
label_encoders = load('../../imputers/encoders.joblib')
X_test = label_encoding_test(X_test, cat_features, label_encoders)

print("Finish label encoding")

# Fill missing categorical values
classifiers = load('../../imputers/classifiers.joblib')
for column, clf in classifiers.items():
    fill_with_trained_imputer(X_test, clf, column, -1)

print("Finish imputing")

# Scaler
scaler = load('../../imputers/scaler.joblib')
X_test[num_features] = scaler.transform(X_test[num_features])

X_test = X_test.values

# Load the model to predict
best_clf = load('../../models/best_clf.pkl')

# Now load y_test and see the values and plot the result
y_test = pd.read_csv('../../test_data/y_test.csv').values[:,0]
print(y_test.shape)
y_test_pred = model_evaluation(best_clf, X_test, y_test)
print(y_test_pred.shape)

residuals = y_test - y_test_pred
histplot(residuals)