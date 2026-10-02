#===========================================================================
# Loading Libraries and env
#===========================================================================
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import random
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

#===========================================================================
# Initialize the LLMs
#===========================================================================
mistral = ChatMistralAI(model= 'open-mistral-7b',temperature= 0.7)

llm_repo_for_huggingface = HuggingFaceEndpoint(
    model = 'meta-llama/Llama-3.1-8B-Instruct',
    temperature= 0.7
)
llama = ChatHuggingFace(llm=llm_repo_for_huggingface)

parser = StrOutputParser()
#===========================================================================
# Loading the vector database and the embedding model
#===========================================================================
embedding_model = HuggingFaceEmbeddings(
    model_name = 'google/embeddinggemma-300m',
    model_kwargs = {'device':'cuda',
                    'local_files_only':True},
    encode_kwargs = {'normalize_embeddings':True}
)

vector_store = Chroma(
    collection_name = 'deep_learning_book',
    embedding_function = embedding_model,
    persist_directory='E:/databases/RAG_project'
)

#===========================================================================
# Retriever
#===========================================================================
retriever = vector_store.as_retriever(search_type = 'mmr',
                                      search_kwargs = {'k': 4,'fetch_k':10,'lambda_mult':0.5},)

#===========================================================================
# Prompt 
#===========================================================================
prompt = ChatPromptTemplate.from_messages([
    ('system',"""
                You are an helpful AI assistant. Use ONLY the provided context to answer the
                question. Be clear and concise. Do NOT give answer out of the provided context.
                If the answer is not present in the document, say : "I can't find the answer."
                """),
    ('human','Context : {context} and Question :{query}')
])

#===========================================================================
# Calling LLM
#===========================================================================
print('**************************************************************')
print("RAG System Initialized !!!!!!!!")
print('**************************************************************')
print("Write 'end' to end the chat.")

while True:
    user_query = input("You : ").strip()
    if user_query != 'end':
        model = random.choice([mistral,llama])

        context_docs = retriever.invoke(user_query)
        context_list = "\n\n".join(doc.page_content for doc in context_docs)
                                   
        chain = prompt | model | parser
        result = chain.invoke({'context':context_list,
                            'query':user_query})
        print(f"AI : {result}")
    else:
        print('**************************************************************')
        print("Thank You.")
        print('**************************************************************')
        break