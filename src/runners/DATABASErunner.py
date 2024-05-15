from .base import Runner

class DATABASErunner(Runner): 
    def __init__(self, *args, **kwargs): 
        """ This is a dummy class as for the moment the sampler handles everything"""
        pass 

    def single_code_run(self, params: dict, run_dir: str):
        pass 
        return None