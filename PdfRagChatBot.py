import os
import shutil
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class PDFQueryBot:
    def __init__(self):
        """
        Initializes the PDFQueryBot with Google Generative AI for embeddings and retrieval.
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.vector_store_path = "pdf_db/faiss_index"
        self.chain = self._get_conversational_chain()
    
    def _get_conversational_chain(self):
        """
        Creates a conversational QA chain.
        """
        prompt_template = """
        Answer the question as detailed as possible from the provided context from single or multiple PDFs.
        Answer in the same language as the context (Hindi if context is Hindi, English if context is English).
        
        Context: {context}
        Question: {question}
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    
    def extract_text_from_pdfs(self, pdf_files):
        """
        Extracts text content from multiple PDFs.
        """
        text = ""
        for pdf in pdf_files:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def split_text_into_chunks(self, text, chunk_size=10000, chunk_overlap=100):
        """
        Splits extracted text into manageable chunks.
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    
    def create_vector_store(self, chunks):
        """
        Generates and saves a FAISS vector store from text chunks.
        """
        if os.path.exists(self.vector_store_path):
            shutil.rmtree(self.vector_store_path)
        vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        vector_store.save_local(self.vector_store_path)
    
    def load_vector_store(self):
        """
        Loads the FAISS vector store.
        """
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError("Vector store not found. Please process PDFs first.")
        return FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
    
    def answer_query(self, query):
        """
        Processes user queries by retrieving relevant context from the vector store.
        """
        vector_store = self.load_vector_store()
        docs = vector_store.similarity_search(query)
        response = self.chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        return response["output_text"]
    
    def process_pdfs(self, pdf_files):
        """
        Processes uploaded PDFs by extracting text, chunking, and storing vectors.
        """
        text = self.extract_text_from_pdfs(pdf_files)
        chunks = self.split_text_into_chunks(text)
        self.create_vector_store(chunks)
        print("PDF Processing Completed. You can now ask queries.")

# Example Usage
if __name__ == "__main__":
    bot = PDFQueryBot()
    pdf_files = [r"C:\Users\DELL\Downloads\Sample_3rd_Project_II.pdf", ]  # Replace with actual file paths
    bot.process_pdfs(pdf_files)
    query = "Give a summary"
    response = bot.answer_query(query)
    print("Response:", response)
