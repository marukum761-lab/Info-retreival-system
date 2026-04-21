import os
from dotenv import load_dotenv

# PDF
from PyPDF2 import PdfReader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Memory + chain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Groq LLM
from langchain_groq import ChatGroq


# =========================
# ENV SETUP
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# =========================
# PDF TEXT EXTRACTION
# =========================
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        reader = PdfReader(pdf)  # FIXED VARIABLE

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text.strip()


# =========================
# CHUNKING
# =========================
def get_text_chunks(text):

    if not text or not isinstance(text, str):
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    # STRICT CLEANING
    cleaned_chunks = [
        str(chunk).strip()
        for chunk in chunks
        if chunk and isinstance(chunk, str) and chunk.strip()
    ]

    return cleaned_chunks


# =========================
# VECTOR STORE (FAISS)
# =========================
def get_vector_store(text_chunks):

    # FINAL SAFETY CHECK
    text_chunks = [
        t for t in text_chunks
        if isinstance(t, str) and t.strip()
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    return vector_store


# =========================
# CONVERSATIONAL CHAIN (GROQ)
# =========================
def get_conversational_chain(vector_store):

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain