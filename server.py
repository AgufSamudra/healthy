from langchain.vectorstores import Chroma
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastapi import FastAPI
import re


# Inisialisasi model Gemini Pro
api_key = "AIzaSyDpEh8S4jo__bjNtJy2hN9cX838FZyF4Ww"
app = FastAPI()

def cleaning_data_before_similarity(input: str) -> str:
    text = input.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

@app.post("/bot/{questions}")
def questions(questions: str):
    # define Embedding with GoogleEmbedding
    persist_directory = "db_V2_noQA"
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                             google_api_key=api_key)  # Use the variable 'api_key'

    # load vector database
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    similarity_questions = vectordb.similarity_search(cleaning_data_before_similarity(questions), k=3)
    
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-pro')

    prompt_template = f"""
        [SYSTEM]
        Anda adalah seorang BOT Kesehatan.
        Minimal 3 paragraph untuk jawaban.
        [END SYSTEM]

        ========================================

        [CONTEXT]
        {similarity_questions}
        [END CONTEXT]

        ========================================

        [QUESTION]
        {questions}
        [END QUESTION]

        OUTPUT: Jawablah QUESTION berdasarkan pengetahuanmu dari CONTEXT. Jika tidak sesuai CONTEXT cukup hanya bilang tidak tau.
        """.lower()

    response = llm.generate_content(prompt_template)
    
    print(f"\n\n[QUESTION]: {questions}")
    print(f"\n\n[CONTEXT]\n{similarity_questions}")
    
    return response.text
