import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLM:
    def __init__(self, model_type="HuggingFaceTB/SmolLM2-135M-Instruct", system_message="You are a helpful AI assistant.") -> None:
        
        torch.random.manual_seed(0)
        
        self.init_chat_state(system_message)
        
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

    def init_chat_state(self, system_message):
        self.chat_state = [{'role': 'assistant', 'content': system_message}]

    def __call__(self, input_text: str) -> str:
        self.chat_state.append({'role': 'user', 'content': input_text})
        output = self.pipe(self.chat_state, **self.generation_args)[0]["generated_text"]
        self.chat_state.append({'role': 'assistant', 'content': output})
        return output
    
