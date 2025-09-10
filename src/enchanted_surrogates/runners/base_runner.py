from abc import ABC, abstractmethod

class Runner(ABC):
    def __init__(self, **kwargs): 
        pass 
    
    @abstractmethod
    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        raise NotImplementedError("Subclasses must implement this method")
