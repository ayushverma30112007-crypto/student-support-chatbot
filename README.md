# 📄 AI-Powered Document Q&A Chatbot (RAG-based)

An intelligent chatbot that lets users upload **any document** — large handbooks, private records, research papers, internal company files, policy documents, etc. — and get instant, accurate answers to specific questions, without having to manually search through hundreds of pages.

🔗 **Live Demo:** _[add your Streamlit Cloud link here after deployment]_

---

## 📌 Problem Statement

Important information is often buried inside large, unstructured documents — student handbooks, private company policies, legal contracts, research reports, internal knowledge bases. Manually searching through such documents for a specific answer is slow and error-prone.

This project solves that using a **Retrieval-Augmented Generation (RAG)** system: users simply upload their document(s) and ask a question in plain English — the chatbot finds the exact relevant section and generates a precise, grounded answer.

---

## 🧠 What is RAG (Retrieval-Augmented Generation)?

A standard AI chatbot only knows what it was trained on — it has no access to your private or specific documents, and it can "hallucinate" (make up) answers when it doesn't know something.

RAG fixes this by combining **information retrieval** with **AI generation**:

1. **Retrieve** — search the user's uploaded document(s) for the most relevant sections related to the question
2. **Augment** — insert those exact sections into the AI's prompt as context
3. **Generate** — the AI answers using only that retrieved context, so the response is accurate and traceable back to the source document

This makes the system reliable even with **large volumes of private, domain-specific, or confidential data** — since nothing needs to be pre-trained into the model; the document itself is the knowledge source, processed at query time.

---

## ⚙️ How It Works — System Workflow

User uploads document(s) — PDF / TXT, any size, any domain
        ↓
Text Chunking — splits large documents into small, overlapping text segments
        ↓
Embedding Generation (MiniLM model) — each chunk is converted into a numerical vector capturing its meaning
        ↓
FAISS Vector Store — all chunk vectors are indexed for fast similarity search
        ↓
User asks a question
        ↓
Similarity Search (Top-K retrieval) — the question is embedded too, and the most relevant chunks are pulled out
        ↓
Groq LLM (Llama 3.3 70B) — retrieved chunks + question sent to Groq's LLM to generate a grounded, document-specific answer
        ↓
Streamlit Chat UI — answer displayed with source references shown to the user

---

## 🛠️ Tech Stack

| Component | Technology Used | Purpose |
|---|---|---|
| **Frontend / UI** | Streamlit | Chat interface, document upload, source viewer |
| **Embedding Model** | Sentence-Transformers (all-MiniLM-L6-v2) | Converts text into vector embeddings locally, no external API needed |
| **Vector Database** | FAISS | Efficient similarity search, scales well even with large documents |
| **LLM (Answer Generation)** | Groq API — Llama 3.3 70B | Generates precise, context-grounded answers at high speed |
| **Document Parsing** | pypdf | Extracts text from PDF documents of any size |
| **Language** | Python 3.x | Core implementation |
| **Deployment** | Streamlit Community Cloud | Free public hosting |
| **Version Control** | Git + GitHub | Source code management |

---

## 📂 Project Structure

document-qa-chatbot/
├── app.py — Streamlit UI + chat logic
├── rag_utils.py — Core RAG pipeline: chunking, embeddings, retrieval
├── requirements.txt — Python dependencies
├── data/ — Default/sample knowledge base
├── .streamlit/secrets.toml — API key (excluded from Git via .gitignore)
├── .gitignore
└── README.md

---

## ✨ Key Features

- 📄 **Works with any document** — not limited to a fixed domain; upload handbooks, reports, private records, or research papers
- 🔒 **Handles large and private/confidential data** — the document is processed locally and only relevant snippets are sent to the LLM, not the entire file
- 💬 **Conversational interface** with persistent chat history
- 🔍 **Source-cited answers** — every response shows exactly which part of the document it was derived from, ensuring transparency and verifiability
- ⚡ **Fast inference** via Groq's LPU-accelerated infrastructure
- 🔄 **Auto re-indexing** when new documents are uploaded
- 🌐 **Free to build, run, and deploy** — no paid infrastructure required

---

## 🚀 Running Locally

Clone the repository:
git clone https://github.com/YOUR_USERNAME/document-qa-chatbot.git
cd document-qa-chatbot

Install dependencies:
pip install -r requirements.txt

Add your Groq API key to .streamlit/secrets.toml:
GROQ_API_KEY = "your_key_here"

Run the app:
python -m streamlit run app.py

Free Groq API key: https://console.groq.com/keys

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**, which builds and hosts the app directly from this GitHub repository. The Groq API key is stored securely via Streamlit's built-in secrets manager and is never exposed in the source code.

---

## 📖 Real-World Use Cases

- Extracting key answers from large private datasets (internal reports, contracts, compliance documents)
- Student/employee handbook and policy Q&A
- Legal or regulatory document analysis
- Research paper and technical documentation search
- Customer support knowledge base assistants

---

## 🎯 Learning Outcomes (Course Project)

This project demonstrates applied understanding of:
- Retrieval-Augmented Generation (RAG) architecture and design
- Vector embeddings and semantic similarity search
- Working with large-scale/private document data efficiently
- Integrating third-party LLM APIs (Groq)
- End-to-end deployment of an AI web application

---

## 👤 Author : Ayush Varma

Built as a course project demonstrating a scalable, document-agnostic RAG system capable of answering questions from large, private, or domain-specific data sources.
