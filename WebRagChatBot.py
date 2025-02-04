# Importing Libraries
import os
import hashlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
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
    This class implements a Retrieval-Augmented Generation (RAG) pipeline that allows you to load, process,
    and retrieve information from documents available on the web. The pipeline leverages Chroma for vector store,
    Google Generative AI for embedding and model interaction, and Langchain's tools for document retrieval and
    chain-based response generation.
    
    Attributes:
        model: The language model used for generating responses.
    """

    def __init__(self,):
        """
        Initializes the WebRAGPipeline with the provided Google API key, setting up the necessary environment
        variable and initializing the language model for generative AI.
        
        Args:
            api_key (str): The API key for Google Cloud, required for interacting with Google Generative AI.
        """
        os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
        os.environ['USER_AGENT'] = "MyWebRAGPipeline/1.0"  # Set the USER_AGENT environment variable
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)


    def load_document(self, url):
        """
        Loads a document from a given URL using the WebBaseLoader from Langchain. This method retrieves the
        content of a webpage and prepares it for further processing.
        
        Args:
            url (str): The URL of the document to be loaded.
        
        Returns:
            list: A list of documents loaded from the URL, or an empty list in case of error.
        """
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            return docs

        except Exception as e:
            print(f"Error loading document: {e}")
            return []


    def create_chunks(self, docs):
        """
        Splits a list of documents into smaller chunks to facilitate processing by the language model.
        The chunks are created using a recursive character-based text splitter, which splits the documents 
        into parts of manageable size.

        Args:
            docs (list): A list of documents to be split into chunks.
        
        Returns:
            list: A list of document chunks after splitting, or an empty list in case of error.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)
            return chunks

        except Exception as e:
            print(f"Error splitting document: {e}")
            return []


    def create_vector_store(self, chunks, url):
        """
        Converts the document chunks into a vector store using Google Generative AI embeddings. This step
        allows efficient retrieval of relevant documents for the RAG pipeline based on user queries.
        The vector store will be saved to a file named after the URL hash to prevent duplication.
        
        Args:
            chunks (list): A list of document chunks to be embedded and stored.
            url (str): The URL of the document, used to create a unique filename for the vector store.
        
        Returns:
            Chroma: A Chroma vector store created from the document chunks, or None in case of error.
        """
        try:
            # Create a unique filename using the hash of the URL
            url_hash = hashlib.md5(url.encode()).hexdigest()

            # Check if the vector store file already exists
            if os.path.exists(f"web_db/{url_hash}"):
                print(f"Vector store already exists for URL: {url}")
                # Load the existing vector store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                vector_store = Chroma(persist_directory=f'web_db/{url_hash}/', embedding_function=embeddings)
                return vector_store
            else:
                print(f"Creating new vector store for URL: {url}")
                # If not exists, create the vector store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=f'web_db/{url_hash}/', )
                return vector_store  # No need to persist manually

        except Exception as e:
            print(f"Error creating or loading vector store: {e}")
            return None


    def get_rag_chain(self, retriever):
        """
        Creates a Retrieval-Augmented Generation (RAG) chain to generate responses using retrieved context.
        This chain connects the document retrieval process with the language model, ensuring that the model
        has access to the relevant context when answering user queries.

        Args:
            retriever: The retriever responsible for fetching relevant documents from the vector store.
        
        Returns:
            chain: The RAG chain for generating responses, or None in case of error.
        
        Working:
            When the RAG Chain is Used:
                1️⃣ User asks a question
                    Example: "What are the latest AI trends?"

                2️⃣ Retriever searches for relevant documents
                    The retriever looks in a vector database (e.g., Chroma, FAISS).
                    It returns relevant documents.

                3️⃣ Retrieved documents are processed by the document chain
                    These documents are passed into document_chain along with the user's question.
               
                4️⃣ Language model generates a response
                    * model receives retrieved context.
                    * user's question.
                    * generates an answer based on both.

                5️⃣ Final response is returned
                    The RAG pipeline outputs the final response.
        """
        try:
            # Define the prompt template for the model
            prompt = ChatPromptTemplate.from_messages([ 
                ("system", "Answer the user's question using the following context:\n\n{context}"), 
                ("user", "{input}"), 
            ])

            # Prepare the document chain for processing retrieved documents
            document_chain = create_stuff_documents_chain(self.model, prompt)

            # Integrate the retriever with the document chain to form the full RAG pipeline
            return create_retrieval_chain(retriever, document_chain)

        except Exception as e:
            print(f"Error creating RAG chain: {e}")
            return None


    def get_response(self, user_input, vector_store):
        """
        Generates a response to a user's query using the RAG pipeline. This method first retrieves relevant
        documents, processes them, and then invokes the language model to generate a response based on the
        context provided by the retrieved documents.

        Args:
            user_input (str): The user's query or question.
            vector_store (Chroma): The vector store used for retrieving relevant documents.
        
        Returns:
            str: The generated response, or an error message if something goes wrong.
        """
        try:
            if vector_store is None:
                return "Error: Vector store is None. Please check if the vector store was created successfully."

            # Create the retriever from the vector store
            retriever = vector_store.as_retriever()
            if retriever is None:
                return "Error: Failed to create retriever."

            # Create the RAG chain using the retriever
            rag_chain = self.get_rag_chain(retriever)
            if rag_chain is None:
                return "Error: Failed to create RAG chain."

            # Invoke the RAG chain to generate a response
            response = rag_chain.invoke({"input": user_input})
            return response.get('answer', "No response generated.")

        except Exception as e:
            return f"Error in response generation: {e}"



if __name__ == "__main__":
    import datetime

    # Initialize the WebRAGPipeline with a Google API key
    rag_pipeline = WebRAGPipeline()
    
    # Load a document from a URL
    # url = "https://www.langchain.com/langgraph"
    url = "https://www.geeksforgeeks.org/introduction-to-arrays-data-structure-and-algorithm-tutorials/"
    print("--------------- Loading Docs ---------------")
    t = datetime.datetime.now()
    docs = rag_pipeline.load_document(url)
    print("Time Taken-", datetime.datetime.now()-t)
    
    # Split the document into smaller chunks
    print("--------------- Creating Chunks ---------------")
    t = datetime.datetime.now()
    chunks = rag_pipeline.create_chunks(docs)
    print("Time Taken-", datetime.datetime.now()-t)

    # Create a vector store using the document chunks
    print("--------------- Creating Vectors ---------------")
    t = datetime.datetime.now()
    vector_store = rag_pipeline.create_vector_store(chunks, url)
    print("Time Taken-", datetime.datetime.now()-t)
    
    # Define a user query to generate a response
    user_input = "What are array"

    # Get the generated response from the RAG pipeline
    print("--------------- Generating Response ---------------")
    t = datetime.datetime.now()
    response = rag_pipeline.get_response(user_input, vector_store)
    print("Time Taken-", datetime.datetime.now()-t)

    # Print the response
    print("\n----------------- Answer -----------------")
    print(response)
    print("----------------- Answer -----------------")
