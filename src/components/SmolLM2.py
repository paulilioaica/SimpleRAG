import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from jinja2 import Template
from src.base.llm_base import LLM

class SmolLM2(LLM):
    def __init__(self, model_type="HuggingFaceTB/SmolLM2-1.7B-Instruct", rag_function=None, example_tools=[]) -> None:
        super(SmolLM2, self).__init__()
        
        torch.random.manual_seed(0)
        
        self.example_tools = [{"search_documents": {
            "description": "Retrieve documents related to a given query term",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The key phrase or search term used to find relevant documents."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of relevant documents to retrieve.",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
        }]


        self.system_message = Template("""You are a helpful AI assitant.
        Your knowledge is limited. If you consider the question to require specific knowledge, you have access to a lot of knowledge in database which you cal access by calling a tool.
        If the given question lacks the parameters required by the function, also point it out.
        You have access to the following tools:
        <tools>{{ tools }}</tools>
        <tool_call> tool format from above </tool_call> Rephrase the question or message by extracting only the key words from it and passing it to the tool.
        If you are passed the CONTEXT, just give use that to output your answer, no need to do calls.""")
        self.init_chat_state(self.system_message)

        
        model = AutoModelForCausalLM.from_pretrained(
            model_type, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_type)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
        )

        self.generation_args = {
            "max_new_tokens":   200,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def init_chat_state(self, system_message):
        self._chat_state = [{'role': 'assistant', 'content': system_message.render(tools=json.dumps(self.example_tools))}]

    def __call_from__rag__(self, context: str) -> str:
        self._chat_state.append({'role': 'user', 'content': context + "\nAbove is relevant context for this question, answer based on this, no need for tool calls\n"})
        output = self.pipe(self._chat_state, **self.generation_args)[0]["generated_text"]
        self._chat_state.append({'role': 'assistant', 'content': output})
        
        return output 
    
    def __call__(self, input_text: str) -> str:
        self._chat_state.append({'role': 'user', 'content': input_text})
        output = self.pipe(self._chat_state, **self.generation_args)[0]["generated_text"]
        self._chat_state.append({'role': 'assistant', 'content': output})
        
        return output
    
