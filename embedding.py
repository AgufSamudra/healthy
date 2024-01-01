import time
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

start_time = time.time()

loader = TextLoader("data/data_sample_10000.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
texts = text_splitter.split_documents(docs)

persist_directory = "db"

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="KEY")

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory= persist_directory)

end_time = time.time()  # Catat waktu selesai
elapsed_time = end_time - start_time

print(f"Proses selesai dalam waktu: {elapsed_time} detik")
