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
* **Hospital**: Akses informasi rumah sakit terdekat berdasarkan lokasi Anda. Fitur Hospital memudahkan Anda dalam menemukan layanan kesehatan di sekitar Anda dengan cepat dan efisien.
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
        * Intim Wanita
        * Menstruasi
        * Keputihan
    
    * Masalah Pencernaan
        * Asam Lambung
        * Sakit Perut
        * Diare
        
    * Pernafasan
        * Asma
        * Batuk
        
    * Gigi dan Mulut
        * Gigi
        * Amandel
    
    * Penyakit Umum
        * Cedera
        * Nyeri Dada
        * Cacar Air
        
    * Alergi
    """)

with col2:
    st.markdown("""
        * Masalah Kulit
            * Kulit
            * Jerawat
            * Gatal
            
        * Kesehatan Anak dan Bayi
            * Bayi
            * Kehamilan
            * Anak
        
        * Obat dan Nutrisi
            * Obat
            * Nutrisi
            
        * Sakit Kepala dan Demam
            * Sakit Kepala
            * Demam
            
        * Masalah Metabolik
            * Diabetes
            * Batu Ginjal
            * Asam Urat
            
        * Intim Laki
        """)
