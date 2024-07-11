from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import MongoDB connection details
MONGODB_URI = "mongodb+srv://law:law123456@clusterone.irlnjku.mongodb.net/?retryWrites=true&w=majority&appName=clusterone"
DB_NAME = "lawdb"
COLLECTION_NAME = "bankruptcy"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_bankruptcy"

# Create FastAPI app
app = FastAPI()

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
llm = LlamaCpp(
        model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        max_tokens=2000,
        top_p=0.1,
        n_ctx=8192,
        n_batch=512,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        handle_parsing_errors=True,
        stream=True,
        verbose=True,
    )

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

# Define endpoint to get response from language model
@app.get("/ask")
async def ask_question(question: str):
    result = qa({"query": question})
    response, sources = process_llm_response(result)
    return {"response": response, "sources": sources}


