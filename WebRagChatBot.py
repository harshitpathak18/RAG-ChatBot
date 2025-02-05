import os
import hashlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

class WebRAGPipeline:
    """
    A pipeline for performing retrieval-augmented generation (RAG) using web data.
    It fetches web content, splits it into chunks, embeds it, and stores the embeddings for retrieval.
    """

    def __init__(self, url):
        """
        Initialize the WebRAGPipeline with a given URL.
        
        Args:
            url (str): The webpage URL to retrieve information from.
        """
        self.url = url
        self.url_hash = hashlib.md5(url.encode()).hexdigest()  # Generate a unique hash for the URL
        
        # Set up environment variables for API keys
        os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
        os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        # Initialize the language model and embedding model
        try:
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        except Exception as e:
            raise RuntimeError(f"Error initializing models: {e}")

        # Load FAISS retriever if exists, otherwise create and store embeddings
        self.retriever = self.load_or_create_retriever()

    def prepare_and_store_embeddings(self):
        """
        Load webpage content, split it into chunks, generate embeddings, and store them locally.
        """
        try:
            loader = WebBaseLoader(self.url)  # Load the web page content
            docs = loader.load()  # Extract text from the web page
            if not docs:
                raise ValueError("No content retrieved from the webpage.")
            
            # Split the text into manageable chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            
            # Create a FAISS vector store from the document chunks
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(f"db_web/{self.url_hash}")  # Save embeddings locally
        except Exception as e:
            raise RuntimeError(f"Error during embedding preparation: {e}")

    def load_or_create_retriever(self):
        """
        Load FAISS retriever if embeddings exist, otherwise generate and store them.
        
        Returns:
            FAISS retriever instance.
        """
        try:
            if not os.path.exists(f'db_web/{self.url_hash}'):
                self.prepare_and_store_embeddings()  # Create embeddings if not found
            
            # Load FAISS retriever for fast lookups
            return FAISS.load_local(f"db_web/{self.url_hash}", self.embeddings, allow_dangerous_deserialization=True).as_retriever()
        except Exception as e:
            raise RuntimeError(f"Error loading retriever: {e}")

    def get_response(self, query):
        """
        Retrieve relevant information from the stored embeddings and generate a response.
        
        Args:
            query (str): The user's query.
        
        Returns:
            str: The generated response based on retrieved data.
        """
        try:
            # Define the prompt template for the language model
            prompt = ChatPromptTemplate.from_messages([ 
                ("system", "Answer the user's question using the following context:\n\n{context}"), 
                ("user", "{input}"), 
            ])

            # Create a document processing chain
            document_chain = create_stuff_documents_chain(self.model, prompt)

            # Create RAG pipeline
            rag_chain = create_retrieval_chain(self.retriever, document_chain)

            # Run the retrieval-augmented generation pipeline
            response = rag_chain.invoke({"input": query})
            
            return response.get('answer', "No response generated.")
        except Exception as e:
            return f"Error generating response: {e}"

if __name__ == "__main__":
    try:
        # Define the target URL
        # url = "https://www.javatpoint.com/machine-learning"
        url = "https://www.javatpoint.com/python-tutorial"
        # url = "https://www.geeksforgeeks.org/introduction-to-arrays-data-structure-and-algorithm-tutorials/"
        
        # Initialize the WebRAGPipeline with the given URL
        rag_pipeline = WebRAGPipeline(url)    

        while True:
            query = input("\nEnter your query (type 'exit' to quit): ")

            if query.lower() == "exit":
                break

            # Generate a response to the user query
            response = rag_pipeline.get_response(query)

            # Print the response
            print("\n----------------- Answer -----------------")
            print(response)
            print("----------------- Answer -----------------")
    
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
