import streamlit as st
import pandas as pd
import numpy as np
from helper import load_model_ml
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

df = pd.read_csv("dataset/9_class/data_to_streamlit.csv")
df.dropna(inplace=True, axis=0)

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

info_text = """
<br>
<div style=padding: 10px; text-align: center; border-radius: 5px;">
    <b>Problem yang bisa di handle</b><br>
    <ul style="list-style-type: disc; padding-left: 20px;">
        <li>Asam Lambung</li>
        <li>Batuk</li>
        <li>Cedera</li>
        <li>Demam</li>
        <li>Gatal</li>
        <li>Gigi</li>
        <li>Jerawat</li>
        <li>Menstruasi</li>
        <li>Sakit Kepala</li>
    </ul>
</div>
<br>
"""


with st.sidebar:
    st.title("Setup")
    st.write("Jika jawaban tidak sesuai bisa memilih 3 model yang di sediakan untuk mendapat jawaban yang sesuai")
    model_selected = st.selectbox("Pilih Model AI yang sesuai",
                 ("Model 1", "Model 2", "Model 3"))
    
    st.markdown(info_text, unsafe_allow_html=True)
    
    st.write("Healthy Bot Version 1.0.1")
    st.write("Thanks for Alo Dokter")
    st.info("Aplikasi ini masih dalam tahap pengembangan. Terima kasih atas pengertian Anda.")

st.title('❤️Healthy Bot')

input_users = st.text_area("Ceritakan Keluhanmu")
input_button = st.button("Submit", type="primary")

if input_button:
    if model_selected == "Model 1":
        
        model = load_model_ml("model/model_1p1_88_85.pkl")
        vector = joblib.load(f"notebook/count_vectorizer.pkl")
        
        input_transform = vector.transform([input_users]) # melakukan transform pada input user
        prediction = model.predict(input_transform)

        filter_data = df[df['kategori']==prediction[0]]

        tf_matrix = vector.transform(filter_data['tanya'])

        cosine_similarities = cosine_similarity(input_transform, tf_matrix)
        most_similar_doc_index = cosine_similarities.argmax()

        most_similar_doc = filter_data.iloc[most_similar_doc_index]
        
        st.markdown(most_similar_doc['jawab'], unsafe_allow_html=True)
    
    if model_selected == "Model 2":
        max_sequence_length = 100

        # muat model dan text vectorizer config
        model = tf.keras.models.load_model("model/model_2p2_90_87.keras")
        vector = tf.keras.layers.TextVectorization(
            max_tokens=5000,
            output_mode='int',
            output_sequence_length=max_sequence_length)

        # mengubah kolom tanya pada DF menjadi numpy array
        training_data = np.array(df['tanya'], dtype=str)

        # mengubah input user menjadi numpy array
        input_array = np.array([input_users], dtype=str)

        # Adaptasikan vector ke Dataframe (training_data)
        vector.adapt(training_data)

        # prediksi input_array (input dari users)
        predictions = model.predict(input_array)

        predictions = np.argmax(predictions)
        filter_data = df[df['kategori'] == predictions]

        # mengubah input_array dan filter_data menjadi vector
        input_vector = vector(input_array).numpy()
        category_questions = vector(filter_data['tanya']).numpy()

        # Hitung cosine similarity antara input pengguna dan semua pertanyaan dalam kategori
        cosine_similarities = cosine_similarity(input_vector, category_questions)
        most_similar_doc_index = cosine_similarities.argmax()

        answer = filter_data.iloc[most_similar_doc_index]['jawab']
        st.markdown(answer, unsafe_allow_html=True)
    
    if model_selected == "Model 3":
        
        model = tf.keras.models.load_model("model/model_3_87_85.h5")
        vector = joblib.load(f"notebook/tokenizer.pkl")
        
        # Sequence
        sequence_testing = vector.texts_to_sequences([input_users])

        # Padding
        max_sequence_length = 200
        padded_testing = tf.keras.preprocessing.sequence.pad_sequences(sequence_testing, 
                                                                        maxlen=max_sequence_length, 
                                                                        padding='post', 
                                                                        truncating='post')
        
        prediction = model.predict(padded_testing).argmax()
        filter_data = df[df['kategori']==prediction]

        sequence_data = vector.texts_to_sequences(filter_data['tanya'])
        padded_data = tf.keras.preprocessing.sequence.pad_sequences(sequence_data, 
                                                                        maxlen=max_sequence_length, 
                                                                        padding='post', 
                                                                        truncating='post')

        cosine_similarities = cosine_similarity(padded_testing, padded_data)
        most_similar_doc_index = cosine_similarities.argmax()

        most_similar_doc = filter_data.iloc[most_similar_doc_index]
        st.markdown(most_similar_doc["jawab"], unsafe_allow_html=True)
        
        