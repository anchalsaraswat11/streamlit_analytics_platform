from dotenv import load_dotenv
load_dotenv()

import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
INDEX_NAME = "policy-index"

def query_policy(question: str) -> str:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )

        docs = vectorstore.similarity_search(question, k=3)

        if not docs:
            return "No relevant policy information found for that question."

        results = []
        for doc in docs:
            results.append(doc.page_content)

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Policy retrieval failed: {str(e)}"


if __name__ == "__main__":
    print(query_policy("What is the return window for electronics?"))