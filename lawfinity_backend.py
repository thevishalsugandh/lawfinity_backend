import os
from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Hardcoded Hugging Face API key
HF_API_KEY = "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"

# MongoDB connection details (replace with your actual values)
MONGODB_URI = "your_mongodb_uri"
DB_NAME = "your_db_name"
COLLECTION_NAME = "your_collection_name"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "your_index_name"
HF_MODEL_URL = "https://api-inference.huggingface.co/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF"

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

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# Define the QA chain
qa_retriever = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
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

# Define endpoint to get response from language model
@app.get("/ask")
async def ask_question(question: str):
    result = qa({"query": question})
    response, sources = process_llm_response(result)
    return {"response": response, "sources": sources}

# Define function to process the response
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    sources = []
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        sources.append(source.metadata['source'])
    return llm_response['result'], sources
