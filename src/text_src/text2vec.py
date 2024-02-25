# This file is to convert text features to vector with BERT
# We will also perform PCA for dimensino reduction
# To save time, I will only perform this on cleaned data

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

def text2vect(tokenizer, bert_model, input_text):
    # Max token is 512
    # We truncate longer input texts
    encoded_input = tokenizer(input_text, return_tensors='tf', max_length=512, truncation=True)
    model_output = bert_model(encoded_input)
    embeddings = model_output.last_hidden_state[:,0,:]
    return embeddings

def concat_vectors_with_pca(df, text_column, tokenizer, bert_model, n_components=10):
    # Generate embeddings
    vectors = np.zeros((len(df), 768))
    for i, text in enumerate(df[text_column]):
        vectors[i, :] = text2vect(tokenizer, bert_model, text)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(vectors)
    
    # Concatenate reduced embeddings with the original DataFrame
    df_copy = df.copy()
    vector_columns = [f'{text_column}_vec_{i}' for i in range(reduced_vectors.shape[1])]
    df_copy_with_vectors = pd.concat([
        df_copy.reset_index(drop=True),
        pd.DataFrame(reduced_vectors, columns=vector_columns),
    ], axis=1)
    
    return df_copy_with_vectors

def transform(df, text_columns, tokenizer, bert_model):
    df_copy = df.copy()
    for col in text_columns:
        df_copy = concat_vectors_with_pca(df_copy,col, tokenizer, bert_model)
    return df_copy