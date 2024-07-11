from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

# Import MongoDB connection details
MONGODB_URI = os.environ.get('MONGODB_URI')
DB_NAME = os.environ.get('DATABASE_NAME')
COLLECTION_NAME = os.environ.get('COLLECTION_NAME')
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.environ.get('ATLAS_VECTOR_SEARCH_INDEX_NAME')

# Create FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create retriever
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

# Load LLM model

llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_length=128,
            temperature=0.01,
            # token="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"
        )

# llm = LlamaCpp(
#         model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#         temperature=0.75,
#         max_tokens=2000,
#         top_p=0.1,
#         n_ctx=8192,
#         n_batch=512,
#         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#         handle_parsing_errors=True,
#         stream=True,
#         verbose=True,
#     )

# Get QA chain
qa_retriever = vector_search.as_retriever(
   search_type="similarity",
   search_kwargs={
       "k": 5
   }
)

prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
   template=prompt_template, input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=qa_retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

# Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    sources = []
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        sources.append(source.metadata['source'])
    return llm_response['result'], sources

@app.get("/")
async def home():
    return {"response": "Hello World"}

class QuestionRequest(BaseModel):
    question: str

# Define endpoint to get response from language model
@app.post("/")
async def ask_question(request: QuestionRequest):
    result = qa({"query": request.question})
    answer, sources = process_llm_response(result)
    return {"answer": answer, "sources": sources}


