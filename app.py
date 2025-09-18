import streamlit as st
from dotenv import load_dotenv
import os
from document_loader import load_and_split_documents
from faiss_vector_store import create_vectorstore, retrieve_relevant_docs
from llm_handler import get_llm, generate_rag_response, generate_pure_llm_response
from chroma_store import ChromaVectorStore
from chat_history import display_chat_history
from private_chat import toggle_private_chat
from export_chat import export_chat_history

load_dotenv()

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'private_mode' not in st.session_state:
    st.session_state.private_mode = False
if 'faiss_vectorstore' not in st.session_state:
    st.session_state.faiss_vectorstore = None
if 'chroma_vector_store' not in st.session_state:
    st.session_state.chroma_vector_store = ChromaVectorStore()
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY")
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = os.getenv("HF_API_KEY")

# Inject Tailwind CSS
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .chat-container { max-width: 800px; margin: auto; }
        .chat-header { background: linear-gradient(to right, #4f46e5, #7c3aed); color: white; padding: 2rem; text-align: center; border-radius: 0.5rem; }
        .chat-message-user { background-color: #e0f2fe; color: #1f2937; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; border: 1px solid #bfdbfe; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chat-message-assistant { background-color: #f3e8ff; color: #4c1d95; padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; border: 1px solid #ddd6fe; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .sidebar { background-color: #f9fafb; padding: 1rem; border-radius: 0.5rem; }
        .btn { background-color: #4f46e5; color: white; padding: 0.5rem 1.5rem; border-radius: 0.25rem; margin: 0.5rem; }
        .btn:hover { background-color: #4338ca; }
        .chat-input { border: 2px solid #4f46e5; border-radius: 0.25rem; padding: 0.5rem; }
        .control-bar { display: flex; justify-content: center; gap: 1rem; padding: 1rem 0; }
        .chat-history { margin-bottom: 1rem; }
        .file-uploader { margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for chat history (from Chroma DB)
with st.sidebar.container():
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-lg font-bold text-gray-800">Stored Chat History</h2>', unsafe_allow_html=True)
    display_chat_history()
    st.markdown('</div>', unsafe_allow_html=True)

# Main UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<div class="chat-header"><h1 class="text-3xl font-bold">Techjays</h1></div>', unsafe_allow_html=True)

# PDF uploader below title
st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
st.markdown('<h2 class="text-lg font-bold text-gray-800">Upload PDFs</h2>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    docs = load_and_split_documents(uploaded_files)
    st.session_state.faiss_vectorstore = create_vectorstore(docs)
    st.success("PDFs processed and FAISS vector store created!")
st.markdown('</div>', unsafe_allow_html=True)

# Control bar for buttons and private chat toggle
st.markdown('<div class="control-bar">', unsafe_allow_html=True)
toggle_private_chat()
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.faiss_vectorstore = None
    st.session_state.chroma_vector_store = ChromaVectorStore()  # Reinitialize to clear
    st.success("Chat history and vector stores cleared!")
export_chat_history()
st.markdown('</div>', unsafe_allow_html=True)

# Display conversation history
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message-user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message-assistant"><strong>AI:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message here...", key="chat_input")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.container():
        st.markdown(f'<div class="chat-message-user"><strong>You:</strong> {user_input}</div>', unsafe_allow_html=True)
    
    # Get LLM and its type
    llm, llm_type = get_llm(st.session_state.gemini_api_key, st.session_state.hf_api_key)
    
    # Retrieve relevant docs if FAISS vectorstore exists
    relevant_docs = []
    if st.session_state.faiss_vectorstore:
        relevant_docs = retrieve_relevant_docs(st.session_state.faiss_vectorstore, user_input)
    
    # Decide RAG or pure LLM
    if relevant_docs:
        response = generate_rag_response(llm, llm_type, user_input, relevant_docs, st.session_state.messages)
    else:
        response = generate_pure_llm_response(llm, llm_type, user_input, st.session_state.messages)
    
    # Display assistant message
    with st.container():
        st.markdown(f'<div class="chat-message-assistant"><strong>AI:</strong> {response}</div>', unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Store in ChromaDB if not private
    if not st.session_state.private_mode:
        st.session_state.chroma_vector_store.add_message(user_input, response)

st.markdown('</div>', unsafe_allow_html=True)