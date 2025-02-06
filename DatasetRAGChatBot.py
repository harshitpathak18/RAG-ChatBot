import os
import pandas as pd
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables from .env file
load_dotenv()

class DatasetRAGPipeline:
    """
    A pipeline for performing RAG using CSV or Excel datasets.
    """

    def __init__(self, dataset_path):
        """
        Initialize with the dataset path, handling CSV or Excel.
        """
        self.dataset_path = dataset_path
        self.df = self._load_dataset(dataset_path)  # Load dataframe during init

        # Set up environment variables for API keys
        os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
        os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        # Initialize the language model and embedding model
        try:
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)
        except Exception as e:
            raise RuntimeError(f"Error initializing models: {e}")

        self.agent = self._create_agent()

    def _load_dataset(self, dataset_path):
        """Loads the dataset, handling CSV and Excel, and encoding."""
        try:
            if dataset_path.endswith('.csv'):
                encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252'] # Common encodings
                
                # First, try without specifying encoding (let pandas infer)
                try:
                    df = pd.read_csv(dataset_path)
                    print("Successfully read CSV without explicit encoding.")
                    return df

                except Exception as e: # Catch any error during inference attempt
                    print(f"Error reading CSV without encoding: {e}")
                    print("Trying common encodings...")
                    
                    # If inference fails, try specific encodings
                    for encoding in encodings_to_try:
                        try:
                            df = pd.read_csv(dataset_path, encoding=encoding)
                            print(f"Successfully read CSV with encoding: {encoding}")
                            return df
                        except UnicodeDecodeError:
                            print(f"Failed to read CSV with encoding: {encoding}")
                            continue  # Try the next encoding
                    raise ValueError(f"Failed to read CSV after trying inference and common encodings: {dataset_path}") # Raise after trying all encodings

            elif dataset_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(dataset_path)
                return df
            else:
                raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(f"Error parsing the dataset: {e}")
        except Exception as e:  # Catch other potential exceptions during file loading
            raise Exception(f"An error occurred while loading the dataset: {e}")


    def _create_agent(self):
        """Creates and returns the LangChain agent."""
        return create_pandas_dataframe_agent(
            llm=self.model,
            df=self.df,
            verbose=False,  # Set to True for debugging
            allow_dangerous_code=True,  # Use with caution!
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )


    def get_response(self, query):
        """Generates a response to a query."""
        return self.agent.invoke(query)


if __name__ == "__main__":
    dataset_path = r"C:\Users\DELL\Downloads\dataset_Facebook.csv"

    try:
        rag_pipeline = DatasetRAGPipeline(dataset_path)

        while True:
            query = input("\nEnter your query (type 'exit' to quit): ")

            if query.lower() == "exit":
                print("Exiting...")
                break

            response = rag_pipeline.get_response(query)

            print("\n----------------- Answer -----------------")
            print(response)
            print("----------------- Answer -----------------")

    except Exception as e:  # Catch and display any exceptions during pipeline setup or query processing
        print(f"An error occurred: {e}")