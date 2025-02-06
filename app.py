import os
import shutil
import pandas as pd
import streamlit as st
from pipeline.WebRag import WebRAGPipeline
from pipeline.PdfRag import PdfRAGPipeline
from pipeline.DatasetRAG import DatasetRAGPipeline
from langchain.memory import ConversationBufferMemory

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

# Initialize session state for chat history and selected RAG source
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "selected_rag" not in st.session_state:
    st.session_state.selected_rag = None

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Sidebar for selecting RAG source
st.sidebar.title("🔍 Select RAG Source")
rag_option = st.sidebar.radio("Choose Source Type", ["Web", "PDF", "Dataset"])

# Web RAG Section
if rag_option == "Web":
    url = st.sidebar.text_input("Enter Website URL")
    if st.sidebar.button("Load Web Data"):
        if url:
            st.session_state.selected_rag = WebRAGPipeline(url)
            st.sidebar.success("✅ Web data processed successfully!")
        else:
            st.sidebar.error("⚠️ Please enter a valid URL.")

# PDF RAG Section
elif rag_option == "PDF":
    pdf_file = st.sidebar.file_uploader("Upload PDF Document", type=["pdf"])
    
    if pdf_file:
        pdf_dir = "uploaded_files/uploaded_pdfs"
        os.makedirs(pdf_dir, exist_ok=True)  # Ensure directory exists

        pdf_path = os.path.join(pdf_dir, pdf_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        st.sidebar.success(f"✅ PDF {pdf_file.name} uploaded successfully!")
        if st.sidebar.button("Process PDF"):
            st.session_state.selected_rag = PdfRAGPipeline(pdf_path)
            st.sidebar.success("✅ PDF processed successfully!")

# Dataset RAG Section
elif rag_option == "Dataset":
    csv_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])
    
    if csv_file:
        dataset_dir = "uploaded_files/uploaded_datasets"
        os.makedirs(dataset_dir, exist_ok=True)  # Ensure directory exists
        
        csv_path = os.path.join(dataset_dir, csv_file.name)
        with open(csv_path, "wb") as f:
            f.write(csv_file.getbuffer())
        
        st.sidebar.success(f"✅ Dataset {csv_file.name} uploaded successfully!")
        
        if st.sidebar.button("Process Dataset"):
            try:
                st.session_state.selected_rag = DatasetRAGPipeline(csv_path)
                st.sidebar.success("✅ Dataset processed successfully!")
            except Exception as e:
                st.sidebar.error(f"⚠️ Error processing dataset: {str(e)}")


# Chat Interface
st.title("💬 RAG Chatbot")
st.write("Ask me anything based on your selected data source!")

# Display chat history
for chat in st.session_state.query_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# Chat input
query = st.chat_input("Type your question...")
if query and st.session_state.selected_rag:
    with st.chat_message("user"):
        st.write(query)

    # Retrieve response
    response = st.session_state.selected_rag.get_response(query)

    with st.chat_message("assistant"):
        st.write(response)

    # Store in memory
    st.session_state.memory.save_context({"input": query}, {"output": response})
    st.session_state.query_history.append({"role": "user", "content": query})
    st.session_state.query_history.append({"role": "assistant", "content": response})
    
elif query and not st.session_state.selected_rag:
    st.sidebar.error("⚠️ Please load a RAG source before asking questions.")
