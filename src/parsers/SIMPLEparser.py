from .base import Parser

class SIMPLEparser(Parser):
    """ An I/O parser for testing """ 
    def __init__(self): 
        pass 

    def write_input_file(self, params: dict, run_dir: str):
        print(run_dir)
        print('Writing to', run_dir)
        