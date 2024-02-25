
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from utilities import model_evaluation, cross_validation, hyper_tune, histplot, plot
from process_train_data import remove_columns, median_fill, fill_totalRent, remove_baseRent, fill_text_features, label_encoding, train_imputer, train_scaler
import joblib
import pickle

df = pd.read_csv('../../data/immo_data.csv')

# Remove outliers
df = df[(df['totalRent'] <= 10000) | pd.isna(df['totalRent'])]
df = df[(df['baseRent'] <= 10000) | pd.isna(df['baseRent'])]
df = df[(df['serviceCharge'] <= 2000) | pd.isna(df['serviceCharge'])]
df = df[(df['floor'] <= 30) | pd.isna(df['floor'])]
df = df[(df['livingSpace'] <= 1000) | pd.isna(df['livingSpace'])]
df = df[(df['noRooms'] <= 10) | pd.isna(df['noRooms'])]

# You can change the random seed
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Create test data
# Remove rows with missing values in totalRent for test
df_test = df_test.dropna(subset=['totalRent'])
# You can also save df_test to see how we process test set
df_test.to_csv('../../test_data/test_data.csv')

# Drop totalRent, baseRent, text features to create X_test
X_test = df_test.drop(columns=['totalRent', 'baseRent', 'description', 'facilities'])
y_test = df_test['totalRent']
X_test.to_csv('../../test_data/X_test.csv', index = False)
y_test.to_csv('../../test_data/y_test.csv', index = False)
print(X_test.shape)
print(y_test.shape)
print(df_test.shape)


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
remove_columns(df_train, columns_1_drop)
remove_columns(df_train, columns_2_drop)
remove_columns(df_train, text_features)

# Median fill by regio2
medians_by_regio2 = median_fill(df_train, num_missing_cols, 'regio2')
# Save medians_by_regio2 DataFrame
medians_by_regio2.to_pickle('../../models/medians_by_regio2.pkl')

# Fill totalRent = baseRent + serviceCharge
df_train = fill_totalRent(df_train)
remove_baseRent(df_train)

# Save y_train and remove totalRent
y_train =df_train['totalRent'].values
df_train.drop(columns =['totalRent'], axis=1, inplace=True)

# Label encoder
df_train, encoders = label_encoding(df_train, cat_features)
joblib.dump(encoders, '../../models/encoders.joblib')

# Fill missing categorical values with Random Forest
classifiers = {}  
for i, col in enumerate(cat_missing_columns):
    clf = train_imputer(df_train, col, -1)
    classifiers[col] = clf
# Save classifiers
joblib.dump(classifiers, '../../models/classifiers.joblib')

# Scaler
df_train, scaler = train_scaler(df_train, num_features)
# Save scaler
joblib.dump(scaler, '../../models/scaler.joblib')

# Ready to define X_train
X_train = df_train.values

# Define model
clf = LGBMRegressor(num_leaves=31)
cross_validation(clf, X_train, y_train)
clf.fit(X_train, y_train)
model_evaluation(clf, X_train, y_train)

param_grid = {
    'num_leaves': [25, 31 ,40],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [80, 100, 120]
}

best_clf = hyper_tune(clf, X_train, y_train, param_grid)

best_clf.fit(X_train, y_train)

y_train_pred = model_evaluation(best_clf, X_train, y_train)

# Plot the residuals
residuals = y_train - y_train_pred
histplot(residuals)
plot(residuals)

# Save the model to disk as a .pkl file
with open('../../models/best_clf.pkl', 'wb') as file:
    pickle.dump(best_clf, file)