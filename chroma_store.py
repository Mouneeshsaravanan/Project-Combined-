import chromadb
from chromadb.utils import embedding_functions

class ChromaVectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="chat_history")
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
    def add_message(self, user_msg, assistant_msg):
        user_embedding = self.embedding_function([user_msg])[0]
        assistant_embedding = self.embedding_function([assistant_msg])[0]
        self.collection.add(
            documents=[user_msg, assistant_msg],
            embeddings=[user_embedding, assistant_embedding],
            ids=[f"user_{len(self.collection.get()['ids'])}", f"assistant_{len(self.collection.get()['ids'])}"],
            metadatas=[{"type": "user"}, {"type": "assistant"}]
        )
    
    def get_all_messages(self):
        results = self.collection.get()
        messages = []
        for i in range(0, len(results['documents']), 2):
            if i+1 < len(results['documents']):
                messages.append({
                    "user": results['documents'][i],
                    "assistant": results['documents'][i+1]
                })
        return messages