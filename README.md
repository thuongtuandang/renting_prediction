# Retal Prediction

This project is to develop machine learning models for predicting retal price in Germany. The data was scraped from Immoscout24, and is available at: 

https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany/data


# Stucture of the project:

The project is organized as follows:

- Notebooks: this folder contains 3 notebook files:
    - EDA_NoTextModels: this notebook is to
            1. perform data analysis
            2. fill missing data
            3. detect and remove outliers
            4. export cleaned data into train, validation and test
            5. Perform a number of machine learning models and results
    - Text2Vect: this notebook is to
        1. use portions of cleaned data to tokenize text features using BERT
        2. perform PCA to reduce the dimension output of BERT from 768 to 10
        3. concat this to cleaned datasets
        4. export data with text vector features
    - TextModels: this notebook is to
        1. Perform prediction with LightGBM on data with text vector features
        2. Perform prediction with saved model (without text) on the same test set
        3. Conclusion


- src: this folder contains .py files to create saved models.

- data: this folder contains:
    - original data (immo_data.csv)
    - cleaned data exported from EDA_NoTextModels notebook:
        1. train_data_no_text, val_data_no_text, test_data_no_text
        2. train_data, val_data, test_data
    - sampling data for Text2Vect: train_data_sample, val_data_sample, test_data_sample  
    - text vectorized sampling data: vect_train_data_sample, vect_val_data_sample, vect_test_data_sample
