from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

#===========================================================================
# Document Loading and Splitting
#===========================================================================
loader = PyMuPDFLoader(file_path='E:/Goal ML/Goal ML/Learnings/Learn_RAG/learnings/deep_learning.pdf')
document = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

raw_chunks = splitter.split_documents(document)

#========================================================================================
# Cleaning the chunks for blank pages, none objects and whitespace only chunks :
#========================================================================================
chunks = [
    chunk for chunk in raw_chunks
        if isinstance(chunk.page_content,str) and chunk.page_content.strip()
]

#========================================================================================
# Create embeddings of chunks and store in vector DB
#========================================================================================
embedding_model = HuggingFaceEmbeddings(
    model_name = 'google/embeddinggemma-300m',
    model_kwargs = {'device':'cuda'},
    encode_kwargs = {'normalize_embeddings':True}
)

vector_store = Chroma.from_documents(
    documents = chunks,
    collection_name = 'deep_learning_book',
    embedding = embedding_model,
    persist_directory= 'E:/databases/RAG_project'
)