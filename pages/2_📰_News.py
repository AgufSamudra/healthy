import streamlit as st
import requests
import webbrowser


st.title("News!")
st.success("Berita selalu update setiap harinya! Cukup tekan tombol Read Article untuk membacanya!")

st.markdown("</br></br>", unsafe_allow_html=True)

get_data = requests.get("https://newsapi.org/v2/top-headlines?country=id&category=health&apiKey=30fdc48a05de4f83a8d9eccc3107bdbc")

if get_data.status_code == 200:
    news_data = get_data.json()
    article = news_data["articles"][:10]

    for i in range(10):
        news_title = article[i]["title"]
        news_publish = article[i]["publishedAt"]
        news_url = article[i]["url"]

        st.markdown(f"##### {news_title}")
        st.text(news_publish)
        button = st.button(f"Read Article {i + 1}", key=f"button_{i + 1}", type="primary")

        if button:
            webbrowser.open(news_url, new=2)

        st.markdown("---")
