from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vectorstore(docs):
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def retrieve_relevant_docs(vectorstore, query, threshold=0.5):
    results = vectorstore.similarity_search_with_score(query, k=5)
    relevant = [doc for doc, score in results if score >= threshold]
    return relevant