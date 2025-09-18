import streamlit as st

def toggle_private_chat():
    st.session_state.private_mode = st.checkbox("Private Chat (not stored in memory)")