import os
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create embeddings using HuggingFace
model_path = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}
print("Creating embeddings")
embeddings = None
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,  
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
except Exception as e:
    print(f"Error creating embeddings: {e}")
    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device':'cpu'}
    #Exit code
    exit(1)

print("Embeddings created")

# Create Pinecone vector store
pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")

index_name = "cve-data"
pc = Pinecone(api_key=pinecone_api_key)
print("Pinecone client created")
pinecone_index = pc.Index(index_name)
db = PineconeStore(index=pinecone_index, embedding=embeddings, text_key="text")

print("Pinecone vector store created")

model = os.environ.get("MODEL", "gemma2:2b")
base_url = os.environ.get("BASE_URL", "http://localhost:11434")

llm = Ollama(
    model=model,
    temperature=0.2,
    base_url=base_url,
)

print("LLM created")

retriever = db.as_retriever()

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of context to answer the question at the end. If the context does not contain the answer, respond with "I don't know." Keep your response as concise as possible.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
