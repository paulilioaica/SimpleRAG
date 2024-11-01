from SmolLM2 import LLM
from embedder import Embedder
from local_vector_db import VectorDatabase
import warnings

class RAG_ChatBot:
    def __init__(self, db_name="vectors.db"):

        self.rag_function = [
            {
                "name": "search_documents",
                "description": "Retrieve documents related to the given query term.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The key phrase or search term to find relevant documents."
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of relevant documents to retrieve",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        self.model = LLM(rag_function=self.rag_function)
        self.embedder = Embedder()

        try:
            self.db = VectorDatabase(db_name=db_name)
        except FileNotFoundError:
            warnings.warn(f"Database file '{db_name}' not found. Creating a new database file.")
            self.db = VectorDatabase(db_name="default.db")

        
    def __call__(self, message):
        output = self.model(message)
        return output

    def search_documents(self, query, num_results):
        embedded_query = self.embedder(query)
        context = self.db.find_most_similar(embedded_query, num_results)
        print(context)
        return context