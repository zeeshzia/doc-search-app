
import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-4-hydrogen"


def embed(texts, task):
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        config = types.EmbedContentConfig(task_type=task)
        result = ai.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=config,
        )
        for item in result.embeddings:
            vectors.append(item.values)
    return vectors


def make_chunks(text, size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def read_file(uploaded):
    pages = []
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded)
        for number, page in enumerate(reader.pages, start=1):
            pages.append((number, page.extract_text() or ""))
    else:
        raw = uploaded.read().decode("utf-8", errors="ignore")
        pages.append((1, raw))
    return pages


def build_index(uploaded):
    docs = []
    metas = []
    for number, page_text in read_file(uploaded):
        for chunk in make_chunks(page_text):
            docs.append(chunk)
            metas.append({"file": uploaded.name, "page": number})
    if not docs:
        return None
    client = chromadb.Client()
    name = "docs_" + str(abs(hash(uploaded.name)))
    collection = client.get_or_create_collection(name)
    ids = ["chunk_" + str(i) for i in range(len(docs))]
    collection.add(
        documents=docs,
        embeddings=embed(docs, "RETRIEVAL_DOCUMENT"),
        metadatas=metas,
        ids=ids,
    )
    return collection


def answer(question, collection):
    q_embedding = embed([question], "RETRIEVAL_QUERY")
    found = collection.query(query_embeddings=q_embedding, n_results=4)
    context = "\n\n".join(found["documents"][0])
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n" + context + "\n\nQuestion: " + question
    )
    reply = ai.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return reply.text, found["metadatas"][0]


st.set_page_config(page_title="Document Search", page_icon="📄")
st.title("📄 Search My Documents")

if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.loaded = None

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded and st.session_state.loaded != uploaded.name:
    with st.spinner("Reading and indexing your document..."):
        st.session_state.collection = build_index(uploaded)
        st.session_state.loaded = uploaded.name
    if st.session_state.collection is None:
        st.error("No text found. Scanned PDFs are not supported yet.")
    else:
        st.success("Ready: " + uploaded.name)

if st.session_state.collection is not None:
    question = st.text_input("Ask a question about your document")
    if question:
        with st.spinner("Thinking..."):
            reply, sources = answer(question, st.session_state.collection)
        st.subheader("Answer")
        st.write(reply)
        st.subheader("Sources")
        for s in sources:
            st.caption(s["file"] + ", page " + str(s["page"]))