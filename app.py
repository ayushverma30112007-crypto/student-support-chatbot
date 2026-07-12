"""
AI Chatbot for Student Support Services — RAG-powered
Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud / Hugging Face Spaces (see README.md)
"""

import os
import glob
import streamlit as st
from groq import Groq
from rag_utils import build_vector_store_from_files

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Student Support Chatbot", page_icon="🎓", layout="centered")
st.title("🎓 AI Chatbot for Student Support Services")
st.caption("Ask about admissions, fees, exams, hostel, or any policy in the knowledge base.")

DATA_DIR = "data"
GROQ_MODEL = "llama-3.3-70b-versatile"  # fast + strong free-tier model on Groq

# ---------------------------------------------------------------------------
# Sidebar: API key + document upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    # Prefer a key from Streamlit secrets (used after deployment); otherwise ask the user
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = ""
    api_key = default_key
    if not api_key:
        st.error("⚠️ API key set nahi hai. Admin ko contact karo.")

    st.divider()
    st.subheader("📄 Knowledge Base")
    st.write("Using the default sample handbook, or upload your own documents below.")

    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True
    )

    rebuild = st.button("🔄 Rebuild knowledge base", use_container_width=True)

    st.divider()
    if st.button("🧹 Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Build / cache the vector store
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing documents...")
def load_default_store():
    files = glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    return build_vector_store_from_files(files)


def build_store_from_uploads(files):
    os.makedirs("uploaded_docs", exist_ok=True)
    paths = []
    for f in files:
        path = os.path.join("uploaded_docs", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
    return build_vector_store_from_files(paths)


if "vector_store" not in st.session_state or rebuild:
    if uploaded_files:
        with st.spinner("Indexing your uploaded documents..."):
            st.session_state.vector_store = build_store_from_uploads(uploaded_files)
        st.sidebar.success(f"Indexed {len(uploaded_files)} uploaded file(s).")
    else:
        st.session_state.vector_store = load_default_store()
        if rebuild:
            st.sidebar.info("Using default sample handbook (no files uploaded).")

vector_store = st.session_state.vector_store

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# RAG answer generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful, friendly student support assistant for a college.
Answer the student's question using ONLY the context provided below.
If the answer isn't in the context, say you don't have that information and suggest
they contact the relevant office (see contact info in the context if available).
Keep answers concise, clear, and warm in tone."""


def generate_answer(question: str, api_key: str) -> tuple[str, list]:
    results = vector_store.search(question, top_k=4)
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in results
    )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return completion.choices[0].message.content, results


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question, e.g. 'What is the minimum attendance required?'")

if question:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = generate_answer(question, api_key)
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
                sources = []
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources used"):
                    for r in sources:
                        st.caption(f"**{r['source']}** (relevance: {r['score']:.2f})")
                        st.text(r["text"][:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
