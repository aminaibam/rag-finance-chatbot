#  Personal Finance RAG Chatbot

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline that answers personal finance questions grounded in a local knowledge base — no hallucinations, no made-up facts.

Built with **LangChain**, **FAISS**, and **OpenAI GPT-4o-mini**.

---

##  Architecture

```
User Question
     │
     ▼
Embedding (OpenAI)
     │
     ▼
FAISS Vector Store ──► Top-K relevant chunks
     │
     ▼
Prompt = Question + Chunks
     │
     ▼
GPT-4o-mini
     │
     ▼
Grounded Answer + Sources
```

---

##  Project Structure

```
rag_finance/
├── data/                  # Your .txt knowledge base documents
│   └── finance_basics.txt
├── src/
│   └── rag_chatbot.py     # Main RAG pipeline
├── faiss_index/           # Auto-generated on first run
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/aminaibam/rag-finance-chatbot.git
cd rag-finance-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Add your documents
Drop any `.txt` files into the `data/` folder. A sample finance knowledge base is already included.

### 5. Run the chatbot
```bash
python src/rag_chatbot.py
```

---

##  Example Session

```
💬 Personal Finance RAG Chatbot
Type your question or 'quit' to exit.

❓ You: What is the 50/30/20 rule?
🤖 Bot: The 50/30/20 rule is a budgeting method where 50% of your after-tax
        income goes to needs, 30% to wants, and 20% to savings and debt repayment.
📄 Sources: finance_basics.txt

❓ You: How much should I have in my emergency fund?
🤖 Bot: Financial experts recommend saving 3 to 6 months of living expenses
        in a high-interest savings account.
📄 Sources: finance_basics.txt
```

---

##  Configuration

You can adjust these parameters in `src/rag_chatbot.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `gpt-4o-mini` | OpenAI model used for generation |
| `CHUNK_SIZE` | `500` | Size of each text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K` | `3` | Number of chunks retrieved per query |

---

##  Add Your Own Documents

Simply drop `.txt` files into the `data/` folder and delete the `faiss_index/` folder to rebuild the index.

```bash
rm -rf faiss_index/
python src/rag_chatbot.py
```

---

##  Tech Stack

- [LangChain](https://www.langchain.com/) — RAG orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [OpenAI API](https://platform.openai.com/) — embeddings + generation
- Python 3.10+

---

##  Author

**Amina Ibrah Amadou** — AI Engineer  
[LinkedIn](https://linkedin.com/in/amina-ibrah-amadou) · [GitHub](https://github.com/aminaibam)
