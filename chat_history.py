import streamlit as st
from chroma_store import ChromaVectorStore

def display_chat_history():
    vector_store = st.session_state.get('chroma_vector_store', ChromaVectorStore())
    history = vector_store.get_all_messages()
    for msg in history:
        st.write(f"**User:** {msg['user']}")
        st.write(f"**Assistant:** {msg['assistant']}")
        st.write("---")