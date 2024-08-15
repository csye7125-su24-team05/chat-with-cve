import os
from pprint import pprint
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Create embeddings using HuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
model_path = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_path,  
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# Create Pinecone vector store
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore

index_name = "cve-data"
pc = Pinecone(api_key="cb92dcfd-09ba-4658-ad65-e0dc657287c8")
pinecone_index = pc.Index(index_name)
db = PineconeStore(index=pinecone_index, embedding=embeddings, text_key="text")


llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key="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"
)

# Create the QA prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

app = FastAPI()


@app.get("/")
async def root():
    return {"Hello": "World"}

@app.get("/search")
async def search(query: str):
    result_val = qa_chain.run(query)
    return {
        "response" : result_val
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    pprint("Port is");
    pprint(os.getenv("PORT"))
    uvicorn.run(app, host="localhost", port=8000)

# Create the retriever and QA chain

