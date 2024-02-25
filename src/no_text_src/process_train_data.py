import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Remove duplicate columns
def remove_columns(df, column_list):
    df.drop(columns = column_list, axis=1, inplace=True)

# Fill columns by median values, return the medians
def median_fill(df_train, column_list, groupby_column):
    # Calculate median values for each column grouped by 'regio2'
    medians = df_train.groupby(groupby_column)[column_list].transform('median')

    # Fill missing values in df_train with these medians
    for col in column_list:
        df_train[col].fillna(medians[col], inplace=True)
    
    # Calculate median values for each column grouped by 'regio2' and store in a new DataFrame
    medians_by_regio2 = df_train.groupby(groupby_column)[column_list].median().reset_index()
    return medians_by_regio2

# Fill totalRent by baseRent + serviceCharge
def fill_totalRent(df):
    df_copy = df.copy()
    df_copy['totalRent'] = df_copy['baseRent'] + df_copy['serviceCharge']
    return df_copy

# Then drop baseRent with remove_columns
def remove_baseRent(df):
    df.drop(columns=['baseRent'], axis=1, inplace=True)

# Fill NaN in text features by description
def fill_text_features(df):
    df_copy = df.copy()
    # Filling missing text values with 'keine Beschreibung' for train set
    df_copy['description'] = df_copy['description'].fillna('keine Beschreibung')
    df_copy['facilities'] = df_copy['facilities'].fillna('keine Beschreibung')
    return df_copy


# Label encoding for categorical features
def label_encoding(df, cat_features):
    df_processed = df.copy()
    encoders = {}  # Store encoders in a dictionary

    for feature in cat_features:
        df_processed[feature] = df_processed[feature].fillna(-1)
        encoder = LabelEncoder()
        # Fit the encoder on non-missing values
        valid_index = df_processed[feature] != -1
        encoder.fit(df_processed.loc[valid_index, feature])
        # Transform all values
        df_processed.loc[valid_index, feature] = encoder.transform(df_processed.loc[valid_index, feature])
        # Store the encoder
        encoders[feature] = encoder
        df_processed[feature] = df_processed[feature].astype(int)

    return df_processed, encoders

# Fill missing values with Random Forest, return the clf
def train_imputer(train_df, column, missing_label):
    # Filter out rows where the target column is missing
    train_data = train_df[train_df[column] != missing_label]
    X_train = train_data.drop(column, axis=1)
    y_train = train_data[column]

    # Prepare data with missing values
    X_missing = train_df[train_df[column] == missing_label].drop(column, axis=1)

    # Initialize and train the classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    predicted_values = clf.predict(X_missing)
    train_df.loc[train_df[column] == missing_label, column] = predicted_values

    print(f"Imputation completed for {column} in training set.")
    return clf

# Scaler for training set, return the scaler
def train_scaler(df, num_features):
    df_processed = df.copy()
    # Initialize the StandardScaler
    train_scaler = StandardScaler()
    # Scale numerical features
    df_processed[num_features] = train_scaler.fit_transform(df_processed[num_features])
    return df_processed, train_scaler

