import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from prompt_tamplate import prompt_template, prompt_template_2
from deep_translator import GoogleTranslator


API_KEY = st.secrets["API_KEY"]

translator_to_indonesian = GoogleTranslator(source='auto', target='id')
translator_to_english = GoogleTranslator(source='auto', target='en')

st.title("Healthy Bot")

st.markdown("</br>", unsafe_allow_html=True)

input_user = st.text_area("Ceritakan Keluhan Anda (dengan detail)", placeholder="Ceritakan Keluhan dengan Detail")

button = st.button("Submit", type="primary")

if button:
    input_user_english = translator_to_english.translate(input_user)

    llm = ChatGoogleGenerativeAI(google_api_key=API_KEY, temperature=0.4, model="gemini-pro", convert_system_message_to_human=True)

    # load database and embedding used
    persist_directory = "db"
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k":10})

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    PROMPT_2 = PromptTemplate(template=prompt_template_2, input_variables=["result"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    chain_2 = LLMChain(llm=llm, prompt=PROMPT_2)
    
    try:
        result = chain(input_user_english)["result"]
        
        translate_response = translator_to_indonesian.translate(result)
        st.markdown(translate_response)
    except:
        st.markdown("Mohon maaf saya tidak tau.")
    
