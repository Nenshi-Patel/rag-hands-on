import base64
import hashlib
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
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

READ_PAGE = """You get a picture of one PDF page, its raw text, and the document's endnotes.
Write down every fact on the page, one fact per line. Do not summarize or skip anything.
- Each line must make sense on its own: say what it is about (product, table, chart).
- Copy numbers and words exactly from the raw text.
- Tables: one line per row, with every value next to its column name.
- Charts: every labelled number and what it stands for.
- Icons and call-outs: write them as normal sentences.
- Footnote markers (small numbers after a claim): add the matching endnote in brackets after that fact.
Text only, no commentary."""

RULES = """Answer using only the passages below.
If the passages do not hold the answer, reply exactly:
I could not find this in the document."""


# ---------------------------------------------------------------------
# Reading
# Gemini looks at every page (picture + raw text) and writes one fact per line.
# Runs once per file; the facts are saved next to the search index.
# ---------------------------------------------------------------------
def page_to_facts(picture, text, endnotes):
    image = base64.b64encode(picture).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": f"{READ_PAGE}\n\nRaw text:\n{text}\n\nEndnotes:\n{endnotes}"},
        {"type": "image_url", "image_url": f"data:image/png;base64,{image}"},
    ])
    # goes to the AI: it reads tables, charts, icons and footnotes off the picture
    return chat_ai.invoke([message]).text


def read_pages(data, saved):
    if saved.exists():
        return json.loads(saved.read_text())

    texts = [page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages]
    endnotes = "\n".join(text for text in texts if "Endnotes" in text)
    pictures = pdfium.PdfDocument(BytesIO(data))

    pages = []
    for n, text in enumerate(texts):
        picture = BytesIO()
        pictures[n].render(scale=2).to_pil().save(picture, "PNG")
        pages.append(page_to_facts(picture.getvalue(), text, endnotes))

    saved.write_text(json.dumps(pages))
    return pages


# ---------------------------------------------------------------------
# Chunking and search
# Runs once per file, then is kept and reused.
# ---------------------------------------------------------------------
def words(text):
    # lowercase, no punctuation: "2018," matches "2018"
    return re.findall(r"\w+", text.lower())


def embed_slowly(chunks, folder):
    # pick up what an earlier run already saved, even if it stopped halfway
    if folder.exists():
        vectors = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    else:
        vectors = FAISS.from_documents(chunks[:BATCH], embeddings)

    for start in range(vectors.index.ntotal, len(chunks), BATCH):
        vectors.save_local(folder)  # keep progress, in case the daily limit stops us
        time.sleep(60)  # free tier: 100 embeddings per minute
        vectors.add_documents(chunks[start:start + BATCH])

    vectors.save_local(folder)
    return vectors


@st.cache_resource(show_spinner="Reading the document (first time takes ~3 minutes)...")
def build_search(data):
    folder = INDEX_FOLDER / hashlib.md5(data).hexdigest()
    INDEX_FOLDER.mkdir(exist_ok=True)
    pages = read_pages(data, folder.with_suffix(".json"))

    # CHUNKING: cut only between lines, so a fact or table row is never split
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0, separators=["\n"])
    chunks = splitter.create_documents(pages, metadatas=[{"page": n} for n in range(1, len(pages) + 1)])

    # HYBRID SEARCH: by meaning (FAISS) and by exact words (BM25), merged by rank
    vectors = embed_slowly(chunks, folder)
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
st.header("Visual Report Q&A")

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
