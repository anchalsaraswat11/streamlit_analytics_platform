from dotenv import load_dotenv
load_dotenv()

import os

# APP_MODE controls where policy retrieval runs:
#   - "live": embeds the query with OpenAI embeddings and searches the hosted
#     Pinecone index (policy-index), built by streamlit/ingest_policy.py.
#   - "demo" (default): embeds locally with a free sentence-transformers model
#     and searches a local FAISS index built from the same policy document.
#     No OpenAI key, no Pinecone account required. The retrieval quality is
#     comparable for this use case — both are dense semantic search over the
#     same source text, just with a different (free vs. hosted) embedding model.

APP_MODE = os.getenv("APP_MODE", "demo")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
INDEX_NAME = "policy-index"

LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_FAISS_PATH = os.path.join(_HERE, "../../local_faiss_index")
POLICY_DOCX_PATH = os.path.join(
    _HERE, "../../Generic E-Commerce Company_ Master Policy Compendium.docx"
)

_local_vectorstore = None  # cached in process after first load/build


def _build_local_vectorstore():
    """
    Chunks and embeds the policy compendium locally with a free
    sentence-transformers model, and builds a FAISS index — the local
    equivalent of what ingest_policy.py does against Pinecone/OpenAI.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)

    docs = Docx2txtLoader(POLICY_DOCX_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    ).split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(LOCAL_FAISS_PATH)
    return vectorstore


def _get_local_vectorstore():
    global _local_vectorstore
    if _local_vectorstore is None:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        if os.path.exists(LOCAL_FAISS_PATH):
            embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
            _local_vectorstore = FAISS.load_local(
                LOCAL_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
            )
        else:
            _local_vectorstore = _build_local_vectorstore()
    return _local_vectorstore


def _query_live(question: str, k: int = 3):
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings
    from pinecone import Pinecone

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    return vectorstore.similarity_search(question, k=k)


def _query_demo(question: str, k: int = 3):
    vectorstore = _get_local_vectorstore()
    return vectorstore.similarity_search(question, k=k)


def query_policy(question: str) -> str:
    try:
        docs = _query_live(question) if APP_MODE == "live" else _query_demo(question)

        if not docs:
            return "No relevant policy information found for that question."

        results = [doc.page_content for doc in docs]
        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Policy retrieval failed ({APP_MODE} mode): {str(e)}"


if __name__ == "__main__":
    print(f"APP_MODE={APP_MODE}")
    print(query_policy("What is the return window for electronics?"))