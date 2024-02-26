__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

import langchain
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from streamlit_chat import message

# from load_docs import load_docs

base_dir = os.path.dirname(os.path.abspath(__file__))


@st.cache_data()
def load_docs():
    documents = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            # pdf_path = "./docs/" + file
            pdf_path = os.path.join(base_dir, "docs", file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith(".docx") or file.endswith(".doc"):
            # doc_path = "./docs/" + file
            doc_path = os.path.join(base_dir, "docs", file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            # text_path = "./docs/" + file
            text_path = os.path.join(base_dir, "docs", file)
            loader = TextLoader(text_path, encoding="utf-8")
            documents.extend(loader.load())
        # elif file.endswith(".csv"):
        #     # csv_path = "./docs/" + file
        #     csv_path = os.path.join(base_dir, "docs", file)
        #     loader = CSVLoader(csv_path)
        #     documents.extend(loader.load())

    return documents


# load_dotenv("../.env")
# api_key = os.environ.get("OPENAI_API_KEY")

api_key = st.secrets["OPENAI_API_KEY"]

persist_vector_directory = "./vector-store/son_vector_store_v6"

## Caching
langchain.llm = SQLiteCache("./cache/langchain.db")

llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# load files
documents = load_docs()

# create a chat history
chat_history = []

# text_splitter = CharacterTextSplitter(
#     chunk_size=1200,
#     chunk_overlap=10,
# )
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=10,
)

docs = text_splitter.split_documents(documents)

if os.path.exists(persist_vector_directory):
    # create vector db
    vectordb = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_vector_directory,
    )
    vectordb.persist()
else:
    # create vector db
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_vector_directory,
    )
    vectordb.persist()

# ConversationalRetrievalChain to get info
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    verbose=False,
)

# === Streamlit front-end ===
# with st.sidebar:
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.write(' ')
    
#     with col2:
#         st.image("aksaray_uni_logo.jpg")
    
#     with col3:
#         st.write(' ')
    
#     st.title("AKSARAY ÜNİVERSİTESİ")
        
c1, c2, c3 = st.columns(3)
with c1:
    st.write("")
    
with c2:
    st.image("aksaray_uni_logo_mid.png")
    
with c3:
    st.write("")
        
d1, d2, d3 = st.columns(3)
with d1:
    st.write("")
    
with d2:
    st.title("ASÜ AI")
    
with d3:
    st.write("")


st.header("Sosyal Bilimler Enstitüsü ile ilgili merak ettiklerinizi sorabilirsiniz...")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_query():
    input_text = st.chat_input("Sorunuzu sorun...")
    return input_text


# retrieve the user input
user_input = get_query()
if user_input:
    result = qa_chain(
        {
            "question": user_input,
            "chat_history": chat_history,
        }
    )
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result["answer"])

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        # message(
        #     st.session_state["past"][i],
        #     is_user=True,
        #     key=str(i) + "_user",
        # )
        # message(st.session_state["generated"][i], key=str(i), avatar_style="")

        with st.chat_message("user", avatar="🎓"):
            st.write(st.session_state["past"][i])
        with st.chat_message(name="assistant", avatar="aksaray_uni_logo.png"):
            st.write(st.session_state["generated"][i])
