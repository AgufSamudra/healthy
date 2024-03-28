import streamlit as st
import requests

st.header("Healthy Bot")

with st.form("my_form"):
    
    placeholder="QNA and Consultation Healthy"
    
    input_users = st.text_area("Input Here", placeholder=placeholder).lower()
    
    button_submit = st.form_submit_button("Submit", type="primary")
    
    if button_submit:
        # Kirim permintaan ke server FastAPI
        response = requests.post(f"http://127.0.0.1:8000/bot/{input_users}")
        st.markdown(response.json())
