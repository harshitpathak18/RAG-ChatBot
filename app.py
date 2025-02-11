import os
import shutil
import pandas as pd
import streamlit as st
from pipeline.WebRag import WebRAGPipeline
from pipeline.PdfRag import PdfRAGPipeline
from pipeline.DatasetRAG import DatasetRAGPipeline
from langchain.memory import ConversationBufferMemory
import time

# Set page configuration with a dark theme - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="QueryMate",
    page_icon="https://cdn-icons-png.flaticon.com/128/18057/18057621.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #9bafd9 !important;
        color: #ffffff;
        padding:0px;
        margin:0px;

    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #9bafd9;
        padding: 2rem 1rem;
    }

    .st-emotion-cache-1e349hp, .st-emotion-cache-qcqlej.ekr3hml8{
        background: linear-gradient(90deg, hsla(236, 100%, 8%, 1) 0%, hsla(211, 100%, 28%, 1) 100%);    
    }
    
    /* Chat message containers */
    .user-message {
        background: linear-gradient(135deg, hsla(220, 67%, 35%, 1) 0%, hsla(221, 43%, 61%, 1) 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        max-width: 50%;
        margin-left: auto;
    }

    
    .assistant-message{
        background: linear-gradient(135deg, hsla(220, 67%, 35%, 1) 0%, hsla(221, 43%, 61%, 1) 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        max-width: 80%;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2962ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1e4bd8;
        transform: translateY(-2px);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: #333333;
        color: white;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
    }
    
    /* Success message styling */
    .success-message {
        padding: 1rem;
        border-radius: 8px;
        background-color: #43a047;
        color: white;
    }
    
    /* Error message styling */
    .error-message {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d32f2f;
        color: white;
    }
    
    /* Current data source indicator */
    .data-source-indicator {
        background-color: #2d2d2d;
        padding: 0.3rem;
        border-radius: 8px;
        margin-bottom: 5px;
        border-left: 4px solid #2962ff;
    }
    
    /* Data preview container */
    .data-preview {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid #4a4a4a;
    }


    /*-----------------------------------------------------*/
    .st-emotion-cache-kgpedg, .st-emotion-cache-1qg05tj.e1jram340, .st-emotion-cache-h4xjwg{
        display: none !important;
    }

    .st-emotion-cache-a6qe2i{
        padding:0px 1rem 0px;
    }

    
    .st-emotion-cache-1pqiyj1{
        padding: 15px 15px 10px;
    }

    .st-emotion-cache-1e349hp{
        padding:10px;
    }

    .st-emotion-cache-gi0tri e121c1cl3{
     display: none !important
    }

    .st-emotion-cache-8ng0p7 {
        gap: 0rem;
    }

    .st-emotion-cache-1cvow4s h1{
        padding: 0.25rem 0px 0rem;
    } 

    .st-emotion-cache-6qob1r.e1dbuyne8{
        background: #0b3866;
    }  
    
</style>
""", unsafe_allow_html=True)


# Initialize session states
if "memory" not in st.session_state:
    st.session_state.memory = {}
if "query_history" not in st.session_state:
    st.session_state.query_history = {}
if "selected_rag" not in st.session_state:
    st.session_state.selected_rag = None
if "current_source" not in st.session_state:
    st.session_state.current_source = None
if "source_data" not in st.session_state:
    st.session_state.source_data = None


# Sidebar with improved styling
with st.sidebar:
    st.write("")
    st.markdown("# 📚 Knowledge Source")
    
    rag_option = st.pills("", ["Web 🌐", "PDF 📝", "Dataset 📊"],  
                  selection_mode='single', 
                  default='Web 🌐')
    
    st.markdown("---")

    # Ensure one option is always selected
    if not rag_option:
        st.error("⚠️ Please select a knowledge source to continue.")

    else:
        # Handling different knowledge sources separately
        if "Web" in rag_option:
            st.markdown("### 🌐 Web Source")
            url = st.text_input("Enter Website URL", placeholder="https://example.com")
            if st.button("🔄 Load Web Data", key="web_load"):
                if url:
                    with st.spinner("Processing website content..."):
                        st.session_state.selected_rag = WebRAGPipeline(url)
                        st.session_state.current_source = f"Web: {url}"
                        st.session_state.source_data = {"type": "web", "url": url}
                        st.session_state.query_history['web'] = []  # Initialize history for Web
                    st.toast("Web data processed successfully!", icon='✅')
                else:
                    st.error("⚠️ Please enter a valid URL")
        
        elif "PDF" in rag_option:
            st.markdown("### 📝 PDF Document")
            pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
            if pdf_file:
                pdf_dir = "uploaded_files/uploaded_pdfs"
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_path = os.path.join(pdf_dir, pdf_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                if st.button("👑 Process PDF", key="pdf_process"):
                    with st.spinner("Processing PDF content..."):
                        st.session_state.selected_rag = PdfRAGPipeline(pdf_path)
                        st.session_state.current_source = f"PDF: {pdf_file.name}"
                        st.session_state.source_data = {"type": "pdf", "filename": pdf_file.name}
                        st.session_state.query_history['pdf'] = []  # Initialize history for PDF
                    st.toast('PDF processed successfully!', icon='✅')
        
        else:
            st.markdown("### 📊 Dataset")
            csv_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"], key="dataset_uploader")
            if csv_file:
                dataset_dir = "uploaded_files/uploaded_datasets"
                os.makedirs(dataset_dir, exist_ok=True)
                csv_path = os.path.join(dataset_dir, csv_file.name)

                with open(csv_path, "wb") as f:
                    f.write(csv_file.getbuffer())
                
                if st.button("📊 Process Dataset", key="dataset_process"):
                    try:
                        with st.spinner("Processing dataset..."):
                            st.session_state.selected_rag = DatasetRAGPipeline(csv_path)
                            df = pd.read_csv(csv_path) if csv_file.name.endswith('.csv') else pd.read_excel(csv_path)
                            st.session_state.current_source = f"Dataset: {csv_file.name}"
                            st.session_state.source_data = {"type": "dataset", "filename": csv_file.name, "preview": df, "shape": df.shape}
                            st.session_state.query_history['dataset'] = []  # Initialize history for Dataset
                        
                        st.toast("Dataset processed successfully!", icon="✅")
                    
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")


# Main chat interface
st.markdown("""
    <div>
        <h1 style='text-align: center;'>QueryMate</h1><br>
    </div>
""", unsafe_allow_html=True)



# Display current data source information
if st.session_state.current_source:
    
    st.info(f'{st.session_state.current_source}')
    
    # Display data preview for datasets
    if st.session_state.source_data and st.session_state.source_data["type"] == "dataset":
        with st.expander("📊 View Dataset Preview"):
            st.markdown(f"Dataset Shape: {st.session_state.source_data['shape'][0]} rows × {st.session_state.source_data['shape'][1]} columns")
            st.dataframe(st.session_state.source_data["preview"].head(), use_container_width=True)

else:
    st.markdown("""
        <div style="padding: 18px; border: 1px solid #ddd; border-radius: 10px;">
            <p style="font-size:17px;">
                <strong>QueryMate</strong> is an AI-powered chatbot that helps you extract insights from various sources,
                including websites, PDFs, and datasets. Simply select a knowledge source from the sidebar
                and start asking questions!
            </p>

           <p> 1️⃣ <strong>Instant Answers, Anytime!</strong> – Get quick and accurate responses from web pages, PDFs, and datasets without the hassle of manual searching. </p> 

           <p> 2️⃣ <strong>Easy & Intuitive</strong> – A simple and user-friendly chat interface that makes interacting with data feel effortless.  </p>

           <p> 3️⃣ <strong>Multi-Purpose</strong> – Whether you're a student, researcher, business analyst, or professional, QueryMate adapts to your needs.</p>  

           <p> 4️⃣ <strong>Saves Time & Effort</strong> – No need to dig through lengthy documents—just ask, and get the insights you need instantly!  </p>

        </div>
    """, unsafe_allow_html=True)



# Chat container with history tracking
chat_container = st.container()
current_source_type = st.session_state.source_data["type"] if st.session_state.source_data else "none"

with chat_container:
    for chat in st.session_state.query_history.get(current_source_type, []):
        if chat["role"]=="user":
            st.markdown(f"""
                <div class="user-message">
                    <img src="https://cdn-icons-png.flaticon.com/128/3059/3059442.png" width="35" height="35" style="margin-right: 10px;">
                    {chat["content"]}    
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div class="assistant-message">
                    <img src="https://cdn-icons-png.flaticon.com/128/11628/11628481.png" width="45" height="45" style="margin-right: 10px;">
                    <br>{chat["content"]}
                </div>
            """, unsafe_allow_html=True)
            


# Chat input
query = st.chat_input(f"💭 Ask your question here... ")


if query and st.session_state.selected_rag:
    # Display user message immediately
    st.markdown(f"""
        <div class="user-message">
            <img src="https://cdn-icons-png.flaticon.com/128/3059/3059442.png" width="35" height="35" style="margin-right: 10px;">
            {query}
        </div>
    """, unsafe_allow_html=True)

    # Add user query to history before generating response
    st.session_state.query_history[current_source_type].append({"role": "user", "content": query})

    # Create a placeholder for the assistant's response
    response_placeholder = st.empty()
    
    # Show a "typing..." message while generating response
    response_placeholder.markdown("""
        <div class="assistant-message">
            <img src="https://cdn-icons-png.flaticon.com/128/11628/11628481.png" width="45" height="45" style="margin-right: 10px;">
            <br><i>Typing...</i>
        </div>
    """, unsafe_allow_html=True)

    # Generate response
    response = st.session_state.selected_rag.get_response(query)
    time.sleep(0.5)  # Simulate typing effect

    # Update placeholder with actual response
    response_placeholder.markdown(f"""
        <div class="assistant-message">
            <img src="https://cdn-icons-png.flaticon.com/128/11628/11628481.png" width="45" height="45" style="margin-right: 10px;">
            <br>{response}
        </div>
    """, unsafe_allow_html=True)

    # Append response to history AFTER rendering
    st.session_state.query_history[current_source_type].append({"role": "assistant", "content": response})


# Default case if no source is selected
elif query:
    st.sidebar.error("⚠️ Please select and process a knowledge source first!")
