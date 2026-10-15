import hashlib
import os
import re
import time
import unicodedata
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

# Parts of the report that only add noise: lists and tables, not sentences
SKIP = {"Summary of Contents", "Contents", "Statistical Annex"}

# Letters the PDF font breaks: "7T_he" -> "The", "/f_inancial" -> "financial"
FIXES = {"7T_h": "Th", "TT_h": "Th", "/T_h": "Th", "T_h": "Th", "/f_i": "fi"}

RULES = """Answer using only the passages below.
Each passage starts with its chapter and section in brackets.
Never mix facts from different regions or chapters.
If the passages do not hold the answer, reply exactly:
I could not find this in the document."""


# ---------------------------------------------------------------------
# Structure
# The PDF's bookmarks say where each chapter and region starts.
# ---------------------------------------------------------------------
def section_of_each_page(reader):
    starts = []
    for mark in reader.outline:
        title = " ".join(mark.title.replace("--", "").split())
        # a box bookmark marks only where the box starts, not where it ends
        if not title.startswith("Box"):
            starts.append((reader.get_destination_page_number(mark), title, "--" in mark.title))
    starts.sort()

    sections, chapter = {}, ""
    for i, (first, title, is_region) in enumerate(starts):
        last = starts[i + 1][0] if i + 1 < len(starts) else len(reader.pages)
        if not is_region:
            chapter = title
        for n in range(first, last):
            sections[n] = f"{chapter} > {title}" if is_region else title
    return sections


# ---------------------------------------------------------------------
# Reading
# Skip the noise, repair broken letters, keep each page's section.
# ---------------------------------------------------------------------
def read_pages(data):
    reader = PdfReader(BytesIO(data))
    sections = section_of_each_page(reader)

    pages, cut_off = [], set()
    for n, page in enumerate(reader.pages):
        section = sections.get(n)
        if section is None or section in SKIP or section in cut_off:
            continue

        # "ﬂ" -> "fl", "ﬁ" -> "fi", then the FIXES table
        text = unicodedata.normalize("NFKC", page.extract_text() or "")
        for broken, fixed in FIXES.items():
            text = text.replace(broken, fixed)

        # citation list: keep the text above it, skip the rest of this section
        references = re.search(r"^\s*References\s*$", text, re.M)
        if references:
            text = text[:references.start()]
            cut_off.add(section)

        text = text.replace("-\n", "-")
        pages.append((" ".join(text.split()), {"page": n + 1, "section": section}))
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


@st.cache_resource(show_spinner="Reading the document (first time takes ~10 minutes)...")
def build_search(data):
    pages = read_pages(data)

    # CHUNKING: cut at sentence ends, and remember the page and section of every piece
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=[". ", " "], keep_separator="end"
    )
    chunks = splitter.create_documents([text for text, _ in pages], [info for _, info in pages])

    # HEADING PATH: "[Ch. 2 Regional Outlooks > South Asia] Growth in SAR is..."
    for chunk in chunks:
        chunk.page_content = f"[{chunk.metadata['section']}] {chunk.page_content}"

    # HYBRID SEARCH: by meaning (FAISS) and by exact words (BM25), merged by rank
    vectors = embed_slowly(chunks, INDEX_FOLDER / hashlib.md5(data).hexdigest())
    by_meaning = vectors.as_retriever(search_kwargs={"k": 10})
    by_words = BM25Retriever.from_documents(chunks, k=10, preprocess_func=words)
    sections = {chunk.metadata["section"] for chunk in chunks}
    return EnsembleRetriever(retrievers=[by_meaning, by_words]), sections


# ---------------------------------------------------------------------
# Answering
# If the question names regions or chapters, keep only chunks from there.
# ---------------------------------------------------------------------
def sections_in_question(question, sections):
    q = question.lower()
    places = [f"Ch. {n}" for n in re.findall(r"chapter (\d+)", q)]
    places += [s.split(" > ")[-1] for s in sections if s.split(" > ")[-1].lower() in q]
    return places


def answer(search, sections, question):
    found = search.invoke(question)

    # ROUTING: fall back to everything if the filter leaves nothing
    places = sections_in_question(question, sections)
    if places:
        found = [c for c in found if any(p in c.metadata["section"] for p in places)] or found

    passages = "\n\n".join(chunk.page_content for chunk in found[:8])
    return chat_ai.invoke(f"{RULES}\n\nPassages:\n{passages}\n\nQuestion: {question}").text


# ---------------------------------------------------------------------
# The page
# Streamlit reruns this file after every click. session_state survives.
# ---------------------------------------------------------------------
st.header("Long Report Q&A")

upload = st.file_uploader("Upload a PDF", type="pdf")
if not upload:
    st.stop()

search, sections = build_search(upload.getvalue())

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, text in st.session_state.chat:
    st.chat_message(role).markdown(text)

question = st.chat_input("Ask about the document")
if question:
    st.chat_message("user").markdown(question)
    with st.spinner("Thinking..."):
        reply = answer(search, sections, question).replace("$", r"\$")
    st.chat_message("assistant").markdown(reply)
    st.session_state.chat += [("user", question), ("assistant", reply)]
