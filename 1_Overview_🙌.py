import streamlit as st

st.title("Overview App!")
st.markdown(
    "Aplikasi ini didesain untuk menjadi sahabat setia dalam menjalani perjalanan kesehatan Anda. Saya berkomitmen untuk "
    "menyajikan solusi terkait kesehatan di sini, meskipun tidak sepenuhnya menyeluruh, namun semaksimal mungkin agar "
    "dapat memberikan bantuan yang signifikan kepada banyak orang. Bersama-sama, mari kita jaga kesehatan dan kesejahteraan kita.")

st.markdown("""
Ada beberapa fitur yang ada pada aplikasi ini, sebagai berikut:
* **Healthy Bot**: Temukan kenyamanan dalam berkonsultasi mengenai kesehatan dengan Healthy Bot. Ajukan keluhan kesehatan Anda, dan dapatkan jawaban langsung dalam waktu singkat.
* **News**: Tetap terinformasi dengan fitur berita kesehatan. Setiap hari, Anda dapat menemukan berita-berita terbaru yang relevan, memastikan Anda selalu mendapatkan informasi terkini untuk kesejahteraan Anda.
""")

st.title("Healthy Bot")

st.markdown(f"""
Berkonsultasilah secara instan dengan Healthy Bot. Cukup sampaikan keluhan atau masalah kesehatan Anda, dan saksikan 
bagaimana bot memberikan solusi dengan cepat!

Ada beberapa penyakit yang bisa di handle oleh Healthy Bot:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    * Kewanitaan
    * Masalah Pencernaan
    * Pernafasan
    * Gigi dan Mulut
    * Penyakit Umum
    * Alergi
    """)

with col2:
    st.markdown("""
        * Masalah Kulit
        * Kesehatan Anak dan Bayi
        * Obat dan Nutrisi
        * Sakit Kepala dan Demam
        * Masalah Metabolik
        * Intim Laki
        """)

st.title("Teknologi")
st.markdown("""
            Teknologi yang digunakan adalah sebagai berikut:
            1. Python
            2. Streamlit (Web UI)
            3. Langchain (Framework for LLM)
            4. ChromaDB (Vector Database)
            5. Google Gemini-Pro (LLM Model)
            6. Google Embedding-001 (Embedding Model)
            7. Deep Translator (for Translate)
            """)