from parsers.SIMPLEparser import SIMPLEparser


class SimplePipelineParser():
    '''
    This is an over simplified case where we are running the same code consecutivly. Usually we would import multiple parsers, one for each code being ran.
    '''
    def __init__(self):
        self.simple_parser = SIMPLEparser()
    
    def simple_out_to_simple_in(self, simple_out_dir, simple_in_dir):
        self.simple_parser.read_output_file(simple_out_dir)
    
    