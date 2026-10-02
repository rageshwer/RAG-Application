import random
import streamlit as st
import os
import tempfile
from config import get_embedding_model,build_vector_store,get_llm,get_context_chunks
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PDF RAG Assistant", )
parser = StrOutputParser()

# ---------------------------------------------------------------- state
@st.cache_resource
def load_embedder():
    return get_embedding_model()

embedder = load_embedder()

@st.cache_resource(show_spinner=False)
def get_vector_store(file_bytes,file_name):
    # Create a temporary file on disk for PyMuPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        path = tmp_file.name
    
    db = build_vector_store(path,embedder)

    if os.path.exists(path):
        os.remove(path)
    return db

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Drop Here")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded:
        with st.spinner("Processing PDF and indexing embeddings..."):
        # Passing raw bytes ensures the cache detects if the file content changes
            vector_db = get_vector_store(uploaded.getvalue(),uploaded.name)
            retriever = vector_db.as_retriever(search_type = 'mmr',
                                search_kwargs = {'k': 4,'fetch_k':10,'lambda_mult':0.5})
            st.success("Vector database ready!")


    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- prompt
prompt = ChatPromptTemplate.from_messages([
('system',"""
            You are an helpful AI assistant. Use ONLY the provided context to answer the
            question. Be clear and concise. Do NOT give answer out of the provided context.
            If the answer is not present in the document, say : "I can't find the answer."
            """),
('human','Context : {context} and Question :{query}')
])
#----------------------------------------------------------------- main
st.title("Dynamic PDF RAG Assistant")

if uploaded is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# chat input
if question := st.chat_input("Ask something about the document"):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        model = get_llm()
        context_chunks = get_context_chunks(retriever,question)
                                    
        chain = prompt | model | parser
        answer = chain.invoke({'context':context_chunks,
                            'query':question})

        st.markdown(answer)
