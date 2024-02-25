import pandas as pd

def label_encoding_test(df, cat_features, encoders):
    df_encoded = df.copy()

    for feature in cat_features:
        # Mapping of known labels to encoded values
        known_labels_map = {label: encoders[feature].transform([label])[0] for label in encoders[feature].classes_}

        # Function to encode a single value
        def encode_value(val):
            if pd.isna(val) or val not in known_labels_map:
                return -1  # Handle unseen labels and NaNs
            else:
                return known_labels_map[val]

        # Apply encoding to the column
        df_encoded[feature] = df[feature].apply(encode_value).astype(int)

    return df_encoded

# Fill with trained imputer
def fill_with_trained_imputer(test_df, clf, column, missing_label):
    # Prepare test data with missing values
    X_missing_test = test_df[test_df[column] == missing_label].drop(column, axis=1)

    # Predict and fill missing values in the test set using the trained classifier
    if not X_missing_test.empty:
        predicted_values_test = clf.predict(X_missing_test)
        test_df.loc[test_df[column] == missing_label, column] = predicted_values_test

    print(f"Imputation completed for {column} in test set.")