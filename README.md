# 🩺 HealthTrace

HealthTrace is a Streamlit chatbot that reads your **medical reports** — blood tests, scan reports, prescriptions — and answers your questions about them in plain language. Built with **LangChain**, **Google Gemini** and **FAISS**.

Upload a report and ask *"is my haemoglobin low?"* or *"what should I eat for this?"* instead of searching every value yourself.

It handles PDFs, Word files, slides, plain text and pictures. Scanned reports with no text inside are read by Gemini's vision, so a phone photo of a lab printout works. You can type your question or speak it.

> Not medical advice. It reads your documents and explains them. Decisions belong with your doctor.

---

## ✨ Features

* **Attach with the paperclip** — upload straight from the chat box, one file or several
* **Many file types** — pdf, docx, pptx, txt, png, jpg, jpeg
* **Reads scans and photos** — a page with no text layer is turned into a picture and read by the AI
* **Ask by voice** — a microphone sits inside the chat box
* **Remembers the chat** — follow-up questions like "and the other one?" work
* **Any language** — ask in Gujarati or Hindi about an English document, and get the answer back in your language
* **Always-on documents** — an optional file in `documents/` is loaded at startup, on top of anything you attach
* **Answers from many files at once** — an image and a PDF are searched together
* **Pays once per file** — reading and embedding are cached, so asking again is free

---

## 🛠 Tech Stack

| Purpose | Tool |
|---|---|
| Web page | Streamlit |
| AI framework | LangChain |
| Chat + vision + speech | Google Gemini (`gemini-3.5-flash-lite`) |
| Embeddings | Google (`gemini-embedding-001`) |
| Vector search | FAISS |
| PDF text | pypdf |
| PDF page to image | pypdfium2 |
| Word, slides, text | unstructured |

---

## 🔄 How it works

```
your files ──► text ──► chunks ──► embeddings ──► FAISS
                                                    │
your question ──────────────► embedding ──► closest chunks
                                                    │
                            chat history + chunks ──► Gemini ──► answer
```

1. **Read** — text comes straight out of normal PDFs and Word files. Scans, photos and audio go to Gemini.
2. **Chunk** — the text is cut into 1000-character pieces with 150 characters of overlap, so no sentence is lost at a join.
3. **Embed** — every piece becomes a list of numbers, so searching works by meaning, not by exact words. This is what lets a Gujarati question find English text.
4. **Search** — your question becomes numbers too, and the closest pieces are pulled out.
5. **Answer** — those pieces plus the whole chat history go to Gemini.

---

## 🚀 Setup

**1. Clone the repository**

```bash
git clone https://github.com/Nenshi-Patel/health-trace-chatbot.git
cd health-trace-chatbot
```

**2. Create a virtual environment** (Python 3.12)

```bash
python3.12 -m venv .venv
```

**3. Install the packages**

```bash
.venv/bin/python -m pip install -r requirements.txt
```

**4. Add your Gemini API key**

```bash
cp .env.example .env
```

Open `.env` and paste your key. Get a free one at [Google AI Studio](https://aistudio.google.com/apikey).

```
GOOGLE_API_KEY=your_key_here
```

**5. Run it**

```bash
.venv/bin/python -m streamlit run health_trace.py
```

---

## 📄 Sample document included

The repo ships with **Complete_Diet_and_Nutrition_Guide.pdf** in `documents/`, so the app works the moment you run it. It answers the question *"I am low on X — what should I eat?"*

| Part | Covers |
|---|---|
| **A — Conditions** | type 2 diabetes, high blood pressure, thyroid, cholesterol, fatty liver, uric acid and gout, PCOS, low haemoglobin, kidney disease |
| **B — Vitamins** | A, B1, B2, B3, B6, B9 (folate), B12, C, D, E, K |
| **C — Minerals** | iron, calcium, zinc, magnesium, iodine, potassium |
| **D — Everyday** | a healthy plate, a sample day, food and medicine interactions, reading labels, when to see a doctor |

Each section lists the foods to eat, the amounts, what blocks absorption, and what to pair it with.

**Try asking:**

* *What should I eat to increase my iron?*
* *I am low on vitamin B12 — which foods help?*
* *What should I avoid with high uric acid?*
* *મારે આયર્ન વધારવા શું ખાવું જોઈએ?*

---

## 📎 Adding your own documents

**Normal way — the paperclip.** Click the `+` in the chat box and pick a file, then ask your
question. Attach several at once if you like. They stay for the whole conversation, so you
attach once and keep asking. Uploads live in memory only and are never written to disk.

**Optional — the `documents/` folder.** A file placed here is loaded when the app starts and
searched with every question, on top of whatever you attach. Its name is never shown, so the
bot simply seems to know the subject. This is where the diet guide above lives. Most people
never need to touch it.

> **Restart the app after changing that folder.** It is read once per session, so a browser
> refresh is not enough. The paperclip needs no restart.

---

## 📊 Report Q&A — GPO Annual Report

`gpo_annual_report.py` is a separate app. Upload one PDF, ask questions, get short answers from that document only.
Built for plain-text reports like the GPO 2018 Annual Report.

```bash
.venv/bin/python -m streamlit run gpo_annual_report.py
```

What it does differently from HealthTrace:

1. **Whole sentences** — line breaks are removed, so chunks are cut at sentence ends (600 characters, 100 overlap).
2. **Hybrid search** — meaning (FAISS) plus exact words (BM25), merged by rank. Numbers and names like "FDsys" match exactly.
3. **Grounded answers** — the AI answers only from the passages, or says "I could not find this in the document."

> **First upload takes ~5 minutes.** The free tier allows 100 embeddings per minute, so pieces are embedded
> 90 at a time with a 1-minute pause. The result is saved in `indexes/`, so the same PDF loads instantly next time.

**Try asking:**

* *How many depository libraries did GPO staff visit in FY 2018?*
* *What new digital system replaced FDsys, and when was it fully deployed?*
* *How much did GPO procure in printing and information products in FY 2018 on behalf of federal agencies?*

---

## 💡 Good to know

* **Free tier limits** — 1000 embedding calls per day, 100 per minute. A 50-page document is roughly 60 chunks, so one index build uses 60.
* **Cost per file** — normal PDFs and Word files cost nothing to read. Scanned pages and images cost one Gemini call each, once, then cached.
* **Not medical advice** — the bot answers from your document. Check anything that matters with a doctor.
