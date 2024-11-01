import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLM:
    def __init__(self, model_type="HuggingFaceTB/SmolLM2-135M-Instruct", rag_function=None) -> None:
        torch.random.manual_seed(0)
        
        self.system_message = "You are a helpful AI assistant."
        self._extract_message = "You are tasked with extracting the topic from a user request to further search for a RAG application. Please return a description of the main topic the user is searching for." 
        self.init_chat_state(self.system_message, rag_function)

        
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
            "max_new_tokens": 200,
            "return_full_text": False,
            'early_stopping': True,
            "temperature": 0.0,
            "do_sample": False,
        }

    def init_chat_state(self, system_message, functions):

        self._chat_state = [{'role': 'assistant', 'content': system_message}] if not functions \
            else [{'function_metadata', functions}, {'role': 'assistant', 'content': system_message}] 

    def _extract_important_topic(self):
        self._extract_topic_chat = [{'role': 'assistant', 'content': self._extract_message}]
        topic = self.pipe(self._extract_topic_chat, **self.generation_args)[0]["generated_text"]
        return topic

    def __call__(self, input_text: str) -> str:
        self._chat_state.append({'role': 'user', 'content': input_text})
        output = self.pipe(self._chat_state, **self.generation_args)[0]["generated_text"]
        self._chat_state.append({'role': 'assistant', 'content': output})
        
        return output
    
