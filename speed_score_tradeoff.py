import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from helper import load_model_ml
from sklearn.metrics import f1_score

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # menghilangkan warning tensorflow ketika run

df = pd.read_csv("dataset/9_class/testing.csv")
df.dropna(inplace=True, axis=0)

df2 = pd.read_csv("dataset/9_class/training.csv") # hanya digunakan untuk TextVectorizer tensorflow
df2.dropna(inplace=True, axis=0)

label = {
    'Asam Lambung': 0,
    'Batuk': 1, 
    'Cedera': 2, 
    'Demam': 3,
    'Gatal': 4, 
    'Gigi': 5, 
    'Jerawat': 6, 
    'Menstruasi': 7, 
    'Sakit Kepala': 8
}

df['kategori'] = df['kategori'].map(label)

## Class One
class ModelOne:
    def __init__(self, df, model: str, vector_path: str):
        self.df = df
        self.model = model
        self.vector_path = vector_path
        
    def fit(self):
        # load model
        model = load_model_ml(self.model)
        
        # load vectorizer
        vector = joblib.load(f"notebook/{self.vector_path}")
        X_test = vector.transform(self.df['tanya'])
        
        # hitung waktu untuk melakukan prediksi
        start_time = time.time()
        prediction = model.predict(X_test)
        end_time = time.time()
        
        time_total = end_time - start_time

        actual = self.df['kategori'].values
        
        f1_score_metric = f1_score(actual, prediction, average='weighted') # jika multiclass tambahkan average='weighted'
        
        print(f"Time: {time_total}")
        print(f'Score: {f1_score_metric*100:.2f}%')
        
        
## Class Two
class ModelTwo:
    def __init__(self, df, model: str, prep: str):
        self.df = df
        self.model = model
        self.prep = prep
        
    def fit(self):
        # load model
        model = tf.keras.models.load_model(self.model)
        
        # setting config TextVectorizer
        max_sequence_length = 100
        tvector = tf.keras.layers.TextVectorization(
                            max_tokens=5000,  # Jumlah token unik yang diizinkan
                            output_mode='int',
                            output_sequence_length=max_sequence_length)
        
        # jika tfidf
        if self.prep == "tfidf":
            vector = joblib.load(f"notebook/tfidf_vectorizer.pkl")
            X_test = vector.transform(self.df['tanya'])
            X_test = X_test.toarray()
        
        # jika text vectorizer
        elif self.prep == "tvector":
            train_adapt = np.array(df2.tanya, dtype=str) # mengubah data training menjadi array
            X_test = np.array(self.df.tanya, dtype=str) # mengubah data testing menjadi array
            tvector.adapt(train_adapt) # adapt/fit hanya ke data training
        
        start_time = time.time()
        prediction = model.predict(X_test, verbose=0) # prediksi
        end_time = time.time()
        time_total = end_time - start_time
        
        actual = self.df['kategori'].values
        
        f1_score_metric = f1_score(actual, np.argmax(prediction, axis=1), average='weighted') # jika multiclass tambahkan average='weighted'
        
        print(f"Time: {time_total}")
        print(f'Score: {f1_score_metric*100:.2f}%')
        
        
## Class Three
class ModelThree:
    def __init__(self, df, model: str):
        self.df = df
        self.model = model
        
    def fit(self):
        
        # load tokenize yang sudah di fit pada saat training
        loaded_tokenizer = joblib.load('notebook/tokenizer.pkl')

        # Sequence
        sequence_testing = loaded_tokenizer.texts_to_sequences(self.df['tanya'])

        # Padding
        max_sequence_length = 200
        padded_testing = tf.keras.preprocessing.sequence.pad_sequences(sequence_testing, 
                                                                       maxlen=max_sequence_length, 
                                                                       padding='post', 
                                                                       truncating='post')
         
         # load model
        model = tf.keras.models.load_model(self.model)
        
        start_time = time.time()
        prediction = model.predict(padded_testing, verbose=0) # prediksi
        end_time = time.time()
        time_total = end_time - start_time
        
        actual = self.df['kategori'].values
        
        f1_score_metric = f1_score(actual, np.argmax(prediction, axis=1), average='weighted') # jika multiclass tambahkan average='weighted'
        
        print(f"Time: {time_total}")
        print(f'Score: {f1_score_metric*100:.2f}%')
        

## Speed Test

# Instance Class
model_1p1 = ModelOne(df, "model/model_1p1_88_85.pkl", "count_vectorizer.pkl")
model_1p2 = ModelOne(df, "model/model_1p2_89_85.pkl", "tfidf_vectorizer.pkl")

model_2p1 = ModelTwo(df, "model/model_2p1_90_86.keras", "tfidf")
model_2p2 = ModelTwo(df, "model/model_2p2_90_87.keras", "tvector")

model_3 = ModelThree(df, "model/model_3_87_85.h5")

# FIT
print("=" * 30)
print("\t","  Model 1")
print("=" * 30)
print("Model 1, Pertama:")
model_1p1.fit()

print("\nModel 1, Kedua:")
model_1p2.fit()

print("\n")

print("=" * 30)
print("\t","  Model 2")
print("=" * 30)
print("Model 2, Pertama:")
model_2p1.fit()

print("\nModel 2, Kedua:")
model_2p2.fit()

print("\n")

print("=" * 30)
print("\t","  Model 3")
print("=" * 30)
model_3.fit()
