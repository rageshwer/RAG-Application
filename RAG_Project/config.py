"""
This config file contains the necessary methods for ease of use
"""
import random
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_mistralai import ChatMistralAI

#============================================================================================

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name = 'embedding_model'
    )
    return embedding_model

def build_vector_store(path:str, embedding_model):
    loader = PyMuPDFLoader(path)
    doc = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(doc)

    vector_store = Chroma.from_documents(chunks,embedding_model)
    return vector_store

#============================================================================================
def get_llm():
    mistral = ChatMistralAI(model= 'open-mistral-7b',temperature= 0.7)

    llm_repo_for_huggingface = HuggingFaceEndpoint(
        model = 'meta-llama/Llama-3.1-8B-Instruct',
        temperature= 0.7
    )
    llama = ChatHuggingFace(llm=llm_repo_for_huggingface)

    return random.choice([mistral,llama])

#============================================================================================
def get_context_chunks(retriever,question):
    context_docs = retriever.invoke(question)
    context_list = "\n\n".join(doc.page_content for doc in context_docs)
    return context_list