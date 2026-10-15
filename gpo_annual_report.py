import hashlib
import os
import re
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

chat_ai = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

INDEX_FOLDER = Path("indexes")
BATCH = 90

RULES = """Answer using only the passages below.
If the passages do not hold the answer, reply exactly:
I could not find this in the document."""


# ---------------------------------------------------------------------
# Reading
# One clean string per page, with whole sentences.
# ---------------------------------------------------------------------
def read_pages(data):
    pages = []
    for page in PdfReader(BytesIO(data)).pages:
        # "build-\nout" becomes "build-out"
        text = (page.extract_text() or "").replace("-\n", "-")
        # every line break becomes a space, so sentences are whole again
        pages.append(" ".join(text.split()))
    return pages


# ---------------------------------------------------------------------
# Chunking and search
# Runs once per file, then is kept and reused.
# ---------------------------------------------------------------------
def words(text):
    # lowercase, no punctuation: "2018," matches "2018"
    return re.findall(r"\w+", text.lower())


def embed_slowly(chunks):
    vectors = FAISS.from_documents(chunks[:BATCH], embeddings)
    for start in range(BATCH, len(chunks), BATCH):
        time.sleep(60)  # free tier: 100 embeddings per minute
        vectors.add_documents(chunks[start:start + BATCH])
    return vectors


@st.cache_resource(show_spinner="Reading the document (first time takes ~5 minutes)...")
def build_search(data):
    pages = read_pages(data)

    # CHUNKING: cut at sentence ends, and remember the page of every piece
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, separators=[". ", " "], keep_separator="end"
    )
    chunks = splitter.create_documents(
        pages, metadatas=[{"page": number} for number in range(1, len(pages) + 1)]
    )

    # HYBRID SEARCH: by meaning (FAISS) and by exact words (BM25), merged by rank
    folder = INDEX_FOLDER / hashlib.md5(data).hexdigest()
    if folder.exists():
        vectors = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    else:
        vectors = embed_slowly(chunks)
        vectors.save_local(folder)

    by_meaning = vectors.as_retriever(search_kwargs={"k": 5})
    by_words = BM25Retriever.from_documents(chunks, k=5, preprocess_func=words)
    return EnsembleRetriever(retrievers=[by_meaning, by_words])


# ---------------------------------------------------------------------
# Answering
# The AI answers only from the passages, or says it could not find it.
# ---------------------------------------------------------------------
def answer(search, question):
    found = search.invoke(question)
    passages = "\n\n".join(chunk.page_content for chunk in found)
    return chat_ai.invoke(f"{RULES}\n\nPassages:\n{passages}\n\nQuestion: {question}").text


# ---------------------------------------------------------------------
# The page
# Streamlit reruns this file after every click. session_state survives.
# ---------------------------------------------------------------------
st.header("Report Q&A")

upload = st.file_uploader("Upload a PDF", type="pdf")
if not upload:
    st.stop()

search = build_search(upload.getvalue())

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, text in st.session_state.chat:
    st.chat_message(role).markdown(text)

question = st.chat_input("Ask about the document")
if question:
    st.chat_message("user").markdown(question)
    with st.spinner("Thinking..."):
        reply = answer(search, question).replace("$", r"\$")
    st.chat_message("assistant").markdown(reply)
    st.session_state.chat += [("user", question), ("assistant", reply)]
