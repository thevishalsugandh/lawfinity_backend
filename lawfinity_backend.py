from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.llms import HuggingFaceEndpoint  # Updated import
from langchain_community.vectorstores import MongoDBAtlasVectorSearch  # Updated import
from langchain_core.callbacks.manager import CallbackManager  # Updated import
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import MongoDB connection details from environment variables
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME')

# Validate environment variables
if not all([MONGODB_URI, DB_NAME, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME]):
    raise ValueError("All environment variables must be set.")

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
    token="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"
)

# Define QA retriever
qa_retriever = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Define prompt template
prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Define QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Process LLM response
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    sources = []
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        sources.append(source.metadata['source'])
    return llm_response['result'], sources

# Define home endpoint
@app.get("/")
async def home():
    return {"response": "Hello World"}

# Define request model
class QuestionRequest(BaseModel):
    question: str

# Define endpoint to get response from language model
@app.post("/")
async def ask_question(request: QuestionRequest):
    result = qa({"query": request.question})
    answer, sources = process_llm_response(result)
    return {"answer": answer, "sources": sources}

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("lawfinity_backend:app", host="0.0.0.0", port=10000)
