class LLM:
    def __init__(self, *args, **kwargs) -> None:
        return NotImplementedError
    
    def init_chat_state(self, *args, **kwargs) -> None:
        return NotImplementedError

    def __call__(self, *args, **kwargs) -> str:
        return NotImplementedError
