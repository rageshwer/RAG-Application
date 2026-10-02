#===========================================================================
# Loading Libraries and env
#===========================================================================
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_classic.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

#===========================================================================
# Initialize the LLMs
#===========================================================================
mistral = ChatMistralAI(model= 'mistral-small-2506',temperature= 0.7)

llm_repo_for_huggingface = HuggingFaceEndpoint(
    model = 'meta-llama/Llama-3.1-8B-Instruct',
    temperature= 0.7
)
llama = ChatHuggingFace(llm=llm_repo_for_huggingface)

#===========================================================================
# Prompt
#===========================================================================
template = ChatPromptTemplate.from_messages([
    ('system',"You are an expert in summarizing the document"),
    ('human',"{document}")
])

prompt = ""

#===========================================================================
# Calling LLM
#===========================================================================
result = llama.invoke(prompt)
print(result.content)