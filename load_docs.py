import os

import streamlit as st
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


@st.cache_data()
def load_docs():
    documents = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            pdf_path = "./docs/" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith(".docx") or file.endswith(".doc"):
            doc_path = "./docs/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            text_path = "./docs/" + file
            loader = TextLoader(text_path, encoding="utf-8")
            documents.extend(loader.load())
        elif file.endswith(".csv"):
            csv_path = "./docs/" + file
            loader = CSVLoader(csv_path)
            documents.extend(loader.load())

    return documents
