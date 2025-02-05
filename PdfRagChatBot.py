import os
import hashlib
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

class PdfRAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline for extracting and querying information from a PDF file
    using FAISS vector store and Google's Gemini-1.5 language model.
    """
    def __init__(self, pdf_path):
        """
        Initializes the PdfRAGPipeline with the given PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file.
        """
        try:
            self.pdf_path = pdf_path
            self.pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()  # Generate a unique hash for storage
            
            # Set up environment variables for API keys
            os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
            os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

            # Initialize the language model and embedding model
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

            # Load or create FAISS retriever
            self.retriever = self.load_or_create_retriever()

        except Exception as e:
            print(f"Error initializing PdfRAGPipeline: {e}")



    def prepare_and_store_embeddings(self):
        """
        Extracts text from the PDF, splits it into chunks, and stores the embeddings in a FAISS vector database.
        """
        try:
            text = ""
            reader = PdfReader(self.pdf_path)
            
            for page in reader.pages:
                text += page.extract_text() or ""

            if not text.strip():
                raise ValueError("No text could be extracted from the PDF.")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
            chunks = text_splitter.split_text(text)

            # Create a FAISS vector store from document chunks
            vector_store = FAISS.from_texts(chunks, self.embeddings)
            vector_store.save_local(f"db_pdf/{self.pdf_hash}")  # Save embeddings locally

        except Exception as e:
            print(f"Error preparing and storing embeddings: {e}")



    def load_or_create_retriever(self):
        """
        Loads an existing FAISS retriever if available, otherwise creates a new one.
        
        Returns:
            FAISS retriever object.
        """
        try:
            if not os.path.exists(f'db_pdf/{self.pdf_hash}'):
                self.prepare_and_store_embeddings()  # Create embeddings if not found
            
            return FAISS.load_local(f"db_pdf/{self.pdf_hash}", self.embeddings, allow_dangerous_deserialization=True).as_retriever()
        
        except Exception as e:
            print(f"Error loading or creating retriever: {e}")
            return None



    def get_response(self, query):
        """
        Generates a response for the given user query using the RAG pipeline.
        
        Args:
            query (str): User input query.
        
        Returns:
            str: Generated response.
        """
        try:
            if not query.strip():
                return "Query cannot be empty."
            
            # Define the prompt template
            prompt = ChatPromptTemplate.from_messages([ 
                ("system", "Answer the user's question as detailed as possible. Answer in the same language as the context (Hindi if context is Hindi, English if context is English) using the following context:\n\n{context}"), 
                ("user", "{input}"), 
            ])
            
            # Create a document processing chain
            document_chain = create_stuff_documents_chain(self.model, prompt)
            
            # Create RAG pipeline
            rag_chain = create_retrieval_chain(self.retriever, document_chain)
            
            # Run the pipeline
            response = rag_chain.invoke({"input": query})
            
            return response.get('answer', "No response generated.")

        except Exception as e:
            return f"Error generating response: {e}"



if __name__ == "__main__":

    try:
        # Define the PDF path
        pdf_path = r"C:\Users\DELL\Downloads\Sample_3rd_Project_II.pdf"

        # Initialize the PdfRAGPipeline with the given PDF file
        rag_pipeline = PdfRAGPipeline(pdf_path)
        
        while True:
            query = input("\nEnter your query (type 'exit' to quit): ")

            if query.lower() == "exit":
                print("Exiting...")
                break

            # Generate a response
            response = rag_pipeline.get_response(query)
            
            print("\n----------------- Answer -----------------")
            print(response)
            print("----------------- Answer -----------------")

    except Exception as e:
        print(f"Unexpected error: {e}")
