"""
RAG Chatbot – Personal Finance
================================
A simple Retrieval-Augmented Generation (RAG) pipeline that answers
personal finance questions based on a local knowledge base of text files.

Stack:
  - LangChain 0.3.x : orchestration 
  - FAISS            : vector store
  - OpenAI           : embeddings + generation (GPT-4o-mini)

Usage:
  python src/rag_chatbot.py
"""

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent.parent / "data"
INDEX_DIR  = Path(__file__).parent.parent / "faiss_index"
MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 3


# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a knowledgeable personal finance assistant.
Answer the question using ONLY the context provided below.
Do not invent or infer information beyond what is given.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# ── Build / Load Vector Store ──────────────────────────────────────────────────

def build_index(api_key: str) -> FAISS:
    """Load documents, split into chunks, embed and store in FAISS."""
    print(" Loading documents from data/ ...")
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in {DATA_DIR}. "
            "Add at least one finance document and try again."
        )

    print(f"   Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"   Split into {len(chunks)} chunks.")

    print(" Generating embeddings ...")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f" Index saved to {INDEX_DIR}/\n")
    return vectorstore


def load_index(api_key: str) -> FAISS:
    """Load an existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


# ── RAG Chain  ──────────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore: FAISS, api_key: str) -> RunnableParallel:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    answer_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    # Returns both the generated answer and the source documents
    return RunnableParallel(
        result=answer_chain,
        source_documents=retriever
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()

    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        print(" Loading existing FAISS index ...")
        vectorstore = load_index(api_key)
    else:
        vectorstore = build_index(api_key)

    chain = build_chain(vectorstore, api_key)

    print("=" * 55)
    print("  Personal Finance RAG Chatbot")
    print("  Type your question or 'quit' to exit.")
    print("=" * 55)

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        result = chain.invoke(question)
        answer = result["result"]
        sources = result.get("source_documents", [])

        print(f"\nBot: {answer}")

        if sources:
            filenames = list({
                Path(doc.metadata.get("source", "unknown")).name
                for doc in sources
            })
            print(f"Sources: {', '.join(filenames)}")


if __name__ == "__main__":
    main()
