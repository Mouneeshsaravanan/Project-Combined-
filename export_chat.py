import streamlit as st
import json

def export_chat_history():
    if st.button("Export Chat"):
        chat_data = {"messages": st.session_state.messages}
        st.download_button(
            label="Download JSON",
            data=json.dumps(chat_data),
            file_name="chat_history.json",
            mime="application/json"
        )