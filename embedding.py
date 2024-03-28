from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from itertools import chain
from api_key import api_key_gemini



# loader = TextLoader("dataset/data_ex_3_cleaning/alodokter/alodokter_article.txt", encoding="utf-8")
# docs = loader.load()

# loader2 = TextLoader("dataset/data_ex_3_cleaning/halodoc/halodoc_article.txt", encoding="utf-8")
# docs2 = loader2.load()

# loader3 = TextLoader("dataset/data_ex_3_cleaning/news_health/news_cnn.txt", encoding="utf-8")
# docs3 = loader3.load()

# loader4 = TextLoader("dataset/data_ex_3_cleaning/qna_alodokter/answers_alodokter.txt", encoding="utf-8")
# docs4 = loader4.load()


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=350)

# texts = text_splitter.split_documents(docs)
# texts2 = text_splitter.split_documents(docs2)
# texts3 = text_splitter.split_documents(docs3)
# texts4 = text_splitter.split_documents(docs4)

# list_data = list(chain(texts, texts2, texts3, texts4))

# len(list_data)

# embeddings = HuggingFaceEmbeddings(model_name="firqaaa/indo-sentence-bert-base")

# persist_directory = "db_indo_bert"
# vectordb = Chroma.from_documents(documents=list_data,
#                                  embedding=embeddings,
#                                  persist_directory= persist_directory)



# ===================================================== V2 =================================================


loader = TextLoader("dataset/data_ex_3_cleaning/alodokter/alodokter_article.txt", encoding="utf-8")
docs = loader.load()

loader2 = TextLoader("dataset/data_ex_3_cleaning/halodoc/halodoc_article.txt", encoding="utf-8")
docs2 = loader2.load()

loader3 = TextLoader("dataset/data_ex_3_cleaning/news_health/news_cnn.txt", encoding="utf-8")
docs3 = loader3.load()

from itertools import chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500, separators=[""])

texts = text_splitter.split_documents(docs)
texts2 = text_splitter.split_documents(docs2)
texts3 = text_splitter.split_documents(docs3)

list_data = list(chain(texts, texts2, texts3))

len(list_data)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key_gemini)

persist_directory = "db_V2_noQA"
vectordb = Chroma.from_documents(documents=list_data,
                                 embedding=embeddings,
                                 persist_directory= persist_directory)

