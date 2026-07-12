# 🎓 AI Chatbot for Student Support Services (RAG + Groq)

A retrieval-augmented chatbot that answers student questions (admissions, fees,
exams, hostel, etc.) using your own documents as the knowledge base.

**How it works:** your documents → split into chunks → converted to embeddings
(local, free model) → stored in FAISS → when a student asks a question, the
most relevant chunks are retrieved and sent to Groq's LLM (Llama 3.3) to
generate a grounded answer.

---

## 1. Run it locally first (recommended before deploying)

```bash
# 1. Unzip / cd into the project folder
cd student-support-chatbot

# 2. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

It will open at `http://localhost:8501`. Paste your Groq API key into the
sidebar (get one free at https://console.groq.com/keys) and start chatting.
A sample student handbook is already included in `data/` so you can test
immediately — or upload your own `.txt`/`.pdf` files from the sidebar.

---

## 2. Deploy it for free (get a public link)

### Option A — Streamlit Community Cloud (easiest, recommended)

1. Create a free GitHub account if you don't have one.
2. Create a new **public** GitHub repo and push this whole folder to it:
   ```bash
   git init
   git add .
   git commit -m "Student support RAG chatbot"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/student-support-chatbot.git
   git push -u origin main
   ```
3. Go to **https://share.streamlit.io** → sign in with GitHub → **"New app"**.
4. Select your repo, branch `main`, main file path `app.py` → **Deploy**.
5. Before or after deploying, add your Groq key as a secret so you don't have
   to paste it every time:
   - In the app dashboard → **Settings → Secrets** → add:
     ```
     GROQ_API_KEY = "your_actual_key_here"
     ```
6. Wait ~2 minutes for the build. You'll get a public link like:
   `https://your-app-name.streamlit.app` — share this with anyone.

### Option B — Hugging Face Spaces (also free, good alternative)

1. Go to https://huggingface.co/new-space
2. Choose **Streamlit** as the SDK, give it a name → Create Space.
3. Upload all project files (`app.py`, `rag_utils.py`, `requirements.txt`,
   `data/`) via the "Files" tab, or push via git like above.
4. Go to **Settings → Repository secrets** → add `GROQ_API_KEY` with your key.
5. The Space builds automatically and gives you a public URL like:
   `https://huggingface.co/spaces/YOUR_USERNAME/student-support-chatbot`

---

## 3. Project structure

```
student-support-chatbot/
├── app.py              # Streamlit UI + chat logic
├── rag_utils.py         # Document loading, chunking, embeddings, FAISS retrieval
├── requirements.txt     # Python dependencies
├── data/
│   └── student_handbook.txt   # Sample knowledge base (replace with your own docs)
└── README.md
```

## 4. Customizing for your own use case

- **Replace the knowledge base**: drop your own `.txt`/`.pdf` files into `data/`
  (or upload them live via the sidebar in the running app).
- **Change the LLM model**: edit `GROQ_MODEL` in `app.py`. Other Groq options
  include `llama-3.1-8b-instant` (faster, smaller) or `mixtral-8x7b-32768`.
- **Tune retrieval**: adjust `chunk_size`/`overlap` in `rag_utils.py`, or
  `top_k` in `app.py`'s `generate_answer()` to retrieve more/fewer chunks.

## 5. For your project report / viva

You can describe this as:
> "A Retrieval-Augmented Generation (RAG) based chatbot built with Sentence-
> Transformers for embeddings, FAISS for vector similarity search, and
> Groq's Llama 3.3 70B for response generation, deployed via Streamlit
> Community Cloud."

That single sentence covers the embedding model, vector store, LLM, and
deployment platform — the four things examiners usually ask about.
