from src.components.SmolLM2 import SmolLM2 as LLM
from src.components.embedder import SentenceEmbedder as Embedder
from src.components.local_vector_db import VectorDatabase
import warnings
import re
import json
import warnings
warnings.filterwarnings("ignore")


class RAG_ChatBot:
    def __init__(self, db_name="src\\data\\vectors.db"):

        self.rag_function = [{
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

    def parse_response(self, text: str) -> str | dict[str, any]:
        """Parses a response from the model, returning either the
        parsed list with the tool calls parsed, or the
        model thought or response if couldn't generate one.

        Args:
            text: Response from the model.
        """
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            print("### INVOKED RAG FUNCTIONALITY ###")
            function_call_params = json.loads(matches[0])
            search_query = function_call_params[-1]['arguments']['query']
            print(f"### SEARCHING FOR `{search_query}`")
            relevant_context = "\n".join([x[1] for x in self.search_documents(search_query, 3)])
            print(f"### GATHERING TOP 3 RELEVANT FOUND")
            rag_answer = self.model.__call_from__rag__(relevant_context)
            print(rag_answer)
            return rag_answer


        return text

    def __call__(self, message):
        response = self.model(message)
        output = self.parse_response(response)
        
        return output

    def search_documents(self, query, num_results):
        embedded_query = self.embedder(query)
        context = self.db.find_most_similar(embedded_query, num_results)
        print(context)
        return context