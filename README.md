# Retal Prediction

This project is to develop machine learning models for predicting retal price in Germany. The data was scraped from Immoscout24, and is available at: 

https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany/data


# Stucture of the project:

The project is organized as follows:

- data: you can download the data from kaggle and put it in the folder. After running Notebooks/EDA_NoTextModels, we will have clean dataset divided into three parts: train, validation and test.

- Notebooks: this folder contains 3 notebook files:
    - EDA_NoTextModels: this notebook is to
        - perform data analysis
        - fill missing data
        -  detect and remove outliers
        - export cleaned data into train, validation and test 
        - ***NOTE***: I filled missing values only on the training set, and later use trained imputers and medians (without using the target variables totalRent and baseRent) to fill in the validation and test set.
        - Perform a number of machine learning models and results
    - Text2Vect: this notebook is to
        - use portions of cleaned data to tokenize text features using BERT
        - perform PCA to reduce the dimension output of BERT from 768 to 10
        - ***NOTE***: the first two steps take more than one hour to run only for 10% dataset.
        - concat this to cleaned datasets
        - export data with text vector features
    - TextModels: this notebook is to
        - Perform prediction with LightGBM on data with text vector features


- src: This is to create model files. There are two parts:
    - no_text_src: I reorganized and performed everything in EDA_NoTextModels. For test, please follow the instructions in the file main.py and test.py in the folder if you wish to use your own X_test.
    - text_src: I reorganized and performed everything in the notebook Text2Vect and TextModels. I used only cleaned datasets exported from EDA_NoTextModels. Please read text_src/main.py and text_src/test.py for more details if you wish to customize your test.

- imputers: items in this folder are created after running no_text_src/main.py. And this folder is to save 
    - classifiers (to fill missing categorical features)
    - encoders (to label encode categorical features)
    - medians_by_regio2 (to fill missing values for numerical features by medians)
    - scaler (for standard scaler)
    all of them are saved when we fill missing values on the training data (check src/no_text_src/main.py) and they will be used to fill missing values in the test set (of course without the target variable). For more details, please check src/no_text_src/test.py.

# Methods for filling missing data

The data contains a lot of missing values as well as duplicated columns. To fill missing numerical features, I used median method, because it is more robust to outliers compared to mean fill and is preferable in datasets with outliers or a skewed distribution.

For categorical features, I used random forest classifiers with training data are known values to predict missing values. Note that during the process, I did not use the target variables (totalRent and baseRent).

# Models for non-text features

1. Linear Regression (highest R^2 score 0.8). The Pearson correlations show strong linear relationships between some features and the target variable, and the linear regression model is a good starting point. It is stable with R^2 scores on the validation and test set are around 0.68. The residuals fluctuate around 0, and their distribution is quite symmetric and bell-shaped. If we hyper-tune to increase degree of polynomial features to 2, the R^2 score increases to 0.8 on the test set.

2. Random Forest (highest R^2 score 0.84). To prevent overfit for the random forest model, I used cross validation technique, and the R^2 scores on the validation and test set are quite stable (0.81) and after hyper-tuning the R^2 score on the test set is 0.84.

3. Light GBM (highest R^2 score 0.9). Similar to Random Forest, LightGBM is a tree-based model, but its running time is very fast compared to XGBoos or Random Forest, because LightGBM employs a leaf-wise (best-first) tree growth algorithm, where it chooses the leaf it believes will yield the best split next, regardless of the level of the tree. To prevent overfit, I also used cross validation, and the R^2 scores on the validation and test set are 0.87. After hyper-tunning, the R^2 score on the test set is 0.89. In src/no_text_src/main.py, when we unify training and validation data, the R^2 score on the test set is more than 0.9.

4. Dense neural network. The final model I tried is a dense neural network with keras. The model has one input layer, one hidden layer and one output layer. The R^2 scores for the validation and test set are 0.79.

5. Conclusion. The best model for this part is LightGBM, and we can extract 10 important features for the totalRent prediction, they are:

['geo_plz', 'livingSpace', 'regio2', 'serviceCharge', 'pricetrend', 'regio3', 'yearConstructed', 'regio1', 'picturecount', 'noRooms']


# Models with text features

There are several ways to handle text features. One way is to use the multi-modal model autogluon from Amazon. However, my current resources do not allow me to perform this model, and that's why I tried another way. 

First, I used the pretrained model BERT for German text to tokenize text features. This step may take a lot of time to perform on the whole dataset. I tried it on 10% dataset and it takes more than 1.5 hours, just to tokenize. 

The training process may require a lot more time, because the output for BERT model is 768 and with two text features, the dimension for our dataset can go up to more than 1500. So, I decided to use PCA for dimension reduction. Because of limited resources, I extract 10 dimensions from PCA for each text feature to create datasets for training. 

To speedup the training process, I used LightGBM again because I think for tabular data, with no trend, no seasonal effects, tree-based models excel compared to neural networks.

The result (only for 10% dataset) is quite good with R^2 score 0.81 on the test set. Especially, we can see that in top 10 important features, there is a vector coefficient from PCA: 

['livingSpace', 'geo_plz', 'serviceCharge', 'pricetrend', 'regio2', 'regio1', 'yearConstructed', 'interiorQual', 'regio3', 'description_vec_4'],

where description_vec_4 are the 4-th PCA component in vectorized description.

# Improvements.

Here are some improvements to increase the model performance.

- Research for the zip code of central area of each city, and then compute the distance from a given zip code of the same city to the center: we are assuming that the totalRent is higher in central area and lower when we go further.

- Translate German to English: by translating German to English, the text model performance can increase, because most accurate LLMs support English way better than German.

- With external resources, we can increase the number of components for PCA. This may increase the perfomance of the model.
