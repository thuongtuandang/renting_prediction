import pandas as pd
import numpy as np
from text2vec import transform
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load dataset
# Note that we will use only certain percentage of the cleaned dataset
# Because tokenizing requires a lot of computational resources
# train, val and test data are exported from Notebooks/EDA_NoTextModels.ipynb

def load_dataset(percentage = 0.1):
    # Change the suitable data paths
    df_train = pd.read_csv('../../data/train_data.csv')
    df_val = pd.read_csv('../../data/val_data.csv')
    df_test = pd.read_csv('../../data/test_data.csv')

    # Create samples with 1% of each DataFrame
    df_train_sample = df_train.sample(frac=percentage, random_state=1)
    df_val_sample = df_val.sample(frac=percentage, random_state=1)
    df_test_sample = df_test.sample(frac=percentage, random_state=1)

    # Save
    df_train_sample.to_csv('../../sample_data_text/train_data_sample.csv', index=False)
    df_val_sample.to_csv('../../sample_data_text/val_data_sample.csv', index=False)
    df_test_sample.to_csv('../../sample_data_text/test_data_sample.csv', index=False)

def create_tokenized_dataset():
    df_train_sample = pd.read_csv('../../sample_data_text/train_data_sample.csv')
    df_val_sample = pd.read_csv('../../sample_data_text/val_data_sample.csv')
    df_test_sample = pd.read_csv('../../sample_data_text/test_data_sample.csv')

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

    # Load pre-trained model (weights)
    bert_model = TFBertModel.from_pretrained('bert-base-german-cased')

    text_columns = ['description', 'facilities']
    print('-------Tokenizing-------')

    # Tokenize text features
    df_train_sample = transform(df_train_sample, text_columns, tokenizer, bert_model)
    df_val_sample = transform(df_val_sample, text_columns, tokenizer, bert_model)
    df_test_sample = transform(df_test_sample, text_columns, tokenizer, bert_model)

    # Drop text features
    df_train_sample.drop(columns=text_columns, axis=1, inplace=True)
    df_val_sample.drop(columns=text_columns, axis=1, inplace=True)
    df_test_sample.drop(columns=text_columns, axis=1, inplace=True)

    print('-------Finish tokenizing-------')

    # Save datasets with tokenized texts
    df_train_sample.to_csv('../../sample_data_text/vect_train_data_sample.csv', index=False)
    df_val_sample.to_csv('../../sample_data_text/vect_val_data_sample.csv', index=False)
    df_test_sample.to_csv('../../sample_data_text/vect_test_data_sample.csv', index=False)