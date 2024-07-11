import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import MongoDB connection details from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")

# Create FastAPI app
app = FastAPI()

# Get embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create retriever
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_URI,
    f"{DB_NAME}.{COLLECTION_NAME}",
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

# Define the Hugging Face model URL and API key from environment variables
HF_API_KEY = os.getenv("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")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF"

headers = {
    "Authorization": f"Bearer {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}",
    "Content-Type": "application/json"
}

# Define the QA chain
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

qa = RetrievalQA.from_chain_type(
    llm=LlamaCpp(
        model_path=HF_MODEL_URL,
        temperature=0.75,
        max_tokens=2000,
        top_p=0.1,
        n_ctx=8192,
        n_batch=512,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        handle_parsing_errors=True,
        stream=True,
        verbose=True,
    ),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    sources = []
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        sources.append(source.metadata['source'])
    return llm_response['result'], sources

# Define endpoint to get response from language model
@app.get("/ask")
async def ask_question(question: str):
    result = qa({"query": question})
    response, sources = process_llm_response(result)
    return {"response": response, "sources": sources}
