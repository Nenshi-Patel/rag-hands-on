import base64
import html
import os
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from unstructured.partition.auto import partition

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"
ALLOWED_FILES = ["pdf", "docx", "pptx", "txt", "png", "jpg", "jpeg"]
BUILT_IN_FOLDER = Path("documents")
FILE_MARK = "[file] "

RULES = """Answer from the document extract.
For questions about our chat, use the earlier turns.
If the extract does not cover it, say so. Never invent numbers.
Always reply in the same language as the question."""

READ_THIS_FILE = ("Write down everything in this file. "
                  "Describe any chart or diagram in one line. "
                  "Text only, no commentary.")

STYLE = """
<style>
.st-key-askbox { position: relative; }
.st-key-askbox [data-testid="stChatInputTextArea"] { padding-right: 98px; }
.st-key-askbox [data-testid="stAudioInput"] {
    position: absolute; right: 60px; bottom: 7px; width: 34px;
}
.st-key-askbox [data-testid="stAudioInput"] svg { width: 24px; height: 24px; }
.st-key-askbox [data-testid="stAudioInput"],
.st-key-askbox [data-testid="stAudioInput"] * {
    background: none !important; border: none !important;
}
.st-key-askbox [data-testid="stAudioInputWaveSurfer"],
.st-key-askbox [data-testid="stAudioInputWaveformTimeCode"] { display: none; }
.filecard {
    display: inline-block; padding: 10px 14px; margin-bottom: 8px;
    border-radius: 12px; background: rgba(255, 255, 255, 0.07);
    font-size: 0.85rem; line-height: 1.45; max-width: 260px; word-break: break-word;
}
.filecard b { display: block; margin-bottom: 14px; opacity: 0.5; font-weight: 600; }
</style>
"""


# ---------------------------------------------------------------------
# Built-in documents
# Files dropped in the documents folder. Always loaded, never shown.
# ---------------------------------------------------------------------
def built_in_documents():
    if not BUILT_IN_FOLDER.is_dir():
        return {}

    return {
        file.name: file.read_bytes()
        for file in sorted(BUILT_IN_FOLDER.iterdir())
        if file.suffix.lstrip(".").lower() in ALLOWED_FILES
    }


# ---------------------------------------------------------------------
# The AI model
# Built once and kept, instead of on every question.
# ---------------------------------------------------------------------
@st.cache_resource
def chat_ai():
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_API_KEY)


# ---------------------------------------------------------------------
# Reading files into text
# Pictures and sound go to the AI. Text files are read on this computer.
# ---------------------------------------------------------------------
def media_to_text(data, mime_type):
    encoded = base64.b64encode(data).decode()

    if mime_type.startswith("image"):
        media = {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded}"}
    else:
        media = {"type": "media", "mime_type": mime_type, "data": encoded}

    instruction = {"type": "text", "text": READ_THIS_FILE}

    # goes to the AI: it looks at the picture, or listens to the sound
    return str(chat_ai().invoke([HumanMessage(content=[instruction, media])]).text)


def pdf_to_text(data):
    written_pages = PdfReader(BytesIO(data)).pages
    picture_pages = pdfium.PdfDocument(BytesIO(data))

    text = []
    for number, page in enumerate(written_pages):
        words = page.extract_text() or ""
        if words.strip():
            text.append(words)
        else:
            # nothing to pull out, so this page is a scan
            picture = BytesIO()
            picture_pages[number].render(scale=2).to_pil().save(picture, "PNG")
            text.append(media_to_text(picture.getvalue(), "image/png"))

    return "\n".join(text)


@st.cache_data(show_spinner=False)
def file_to_text(name, data):
    if name.endswith(".png"):
        return media_to_text(data, "image/png")

    if name.endswith((".jpg", ".jpeg")):
        return media_to_text(data, "image/jpeg")

    if name.endswith(".pdf"):
        return pdf_to_text(data)

    parts = partition(file=BytesIO(data), metadata_filename=name)
    return "\n".join(part.text or "" for part in parts)


# ---------------------------------------------------------------------
# Chunking and embedding
# Runs once per set of files, then is kept and reused.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_search_index(names, files):
    text = "\n".join(file_to_text(name, data) for name, data in zip(names, files))
    if not text.strip():
        return None

    # CHUNKING: cut the long text into overlapping pieces
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)

    # EMBEDDING: turn every piece into numbers, so we can search by meaning
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_texts(chunks, embeddings)


# ---------------------------------------------------------------------
# Searching and answering
# ---------------------------------------------------------------------
def answer(question):
    if not question:
        return "Files read. Ask me anything about them."

    documents = st.session_state.documents
    index = build_search_index(tuple(documents), tuple(documents.values()))
    if index is None:
        return "I could not read any text from those files."

    # SEARCH: the question becomes numbers too, then we take the closest pieces
    extract = "\n\n".join(
        match.page_content for match in index.similarity_search(question)
    )

    # the whole chat is sent, but the document pieces ride on this question only
    conversation = st.session_state.history[:-1] + [
        HumanMessage(content=f"Extract:\n{extract}\n\nQuestion: {question}")
    ]

    # goes to the AI: it writes the reply
    return str(chat_ai().invoke(conversation).text)


# ---------------------------------------------------------------------
# Showing messages
# Lines marked with FILE_MARK are drawn as file cards.
# ---------------------------------------------------------------------
def show_question(content):
    blocks = []
    for line in content.split("\n"):
        if line.startswith(FILE_MARK):
            name = html.escape(line[len(FILE_MARK):])
            blocks.append(
                f'<div class="filecard"><b>{name.rsplit(".", 1)[-1].upper()}</b>{name}</div>'
            )
        elif line.strip():
            blocks.append(f"<div>{html.escape(line)}</div>")

    st.markdown("".join(blocks), unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Page setup and memory
# Streamlit reruns this file after every click. session_state survives.
# ---------------------------------------------------------------------
st.header("Health Trace Chatbot")
st.markdown(STYLE, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content=RULES)]
if "documents" not in st.session_state:
    st.session_state.documents = built_in_documents()
if "voice_round" not in st.session_state:
    st.session_state.voice_round = 0


# ---------------------------------------------------------------------
# The chat so far
# Item 0 is the rules, meant for the AI, so it is skipped.
# ---------------------------------------------------------------------
for message in st.session_state.history[1:]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            show_question(message.content)
    else:
        st.chat_message("assistant").markdown(message.content)

asked_now = st.container()


# ---------------------------------------------------------------------
# The input box
# Paperclip, microphone and send arrow all sit inside it.
# ---------------------------------------------------------------------
with st.container(key="askbox"):
    typed = st.chat_input(
        "Ask about your documents", accept_file="multiple", file_type=ALLOWED_FILES
    )
    recording = st.audio_input(
        "voice", key=f"voice_{st.session_state.voice_round}", label_visibility="collapsed"
    )

loader = st.empty()


# ---------------------------------------------------------------------
# When something is sent
# ---------------------------------------------------------------------
if typed or recording:
    uploads = typed.files if typed else []
    for upload in uploads:
        st.session_state.documents[upload.name] = upload.getvalue()

    if recording:
        # a new number gives the recorder a fresh start after this turn
        st.session_state.voice_round += 1
        with loader, st.spinner("Listening..."):
            question = media_to_text(recording.getvalue(), "audio/wav")
    else:
        question = typed.text

    file_lines = [FILE_MARK + upload.name for upload in uploads]
    user_message = "\n".join(file_lines + ([question] if question else []))
    st.session_state.history.append(HumanMessage(content=user_message))
    with asked_now, st.chat_message("user"):
        show_question(user_message)

    with loader, st.spinner("Thinking..."):
        reply = answer(question)

    st.session_state.history.append(AIMessage(content=reply))

    # redraw from the top, so the recorder resets and nothing is drawn twice
    st.rerun()
