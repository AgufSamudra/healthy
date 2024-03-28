from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain

from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent


api_key = "AIzaSyDpEh8S4jo__bjNtJy2hN9cX838FZyF4Ww"

llm = ChatGoogleGenerativeAI(google_api_key=api_key, temperature=0.8, model="gemini-pro", convert_system_message_to_human=True)

# load database and embedding used
persist_directory = "D:/codingproject/healthybot/db"
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyDpEh8S4jo__bjNtJy2hN9cX838FZyF4Ww")

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

from langchain.chains import ConversationalRetrievalChain

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    memory=memory
)

query = """
ello, thank you for your question to Alodokter.<br><br><a href=""https://www.alodokter.com/terkena-mesin-vagina- saat-hamil-whether-berdanga"">On conditions Vaginal discharge during pregnancy that is anything other than clear and greenish in color could indicate an infection in the vagina that needs to be treated immediately during pregnancy to prevent complications such as death of the baby in the womb, low birth weight, birth defects or premature birth. .</p>
<p>Using betel leaf water or feminine cleansing soap is not recommended for use on the vagina, because it can damage or eliminate the good bacteria in the vagina which actually functions to protect the vagina from infection, so that vaginal infections can get worse later.</p>
<p>Therefore, you should not use betel leaf water again, <a href=""https://www.alodokter.com/ketahui-beda-probiotic-dan-prebiotic-and-cepat-both"">food consumption high in prebiotics and probiotics</a>to increase the number of good bacteria in the vagina which can overcome vaginal infections, maintain vaginal cleanliness by regularly changing underwear that is not tight at least 2 times a day or when it is damp, dry the genitals after bathing and defecating, and punch from front to back.</p>
<p>However, you should still check yourself further with <a href=""https://www.alodokter.com/cari-dokter/dokter-kandungan?page=1"">gy

"""
result = conversation_chain({"question": query})
answer = result["answer"]
print(answer)