from .base import Parser


class STATICPOOLparser(Parser):
    """ This is a dummy class as for the moment the sampler handles everything"""
    def __init__(self, *args, **kwargs): 
        self.target_keys = kwargs.get('target')
        self.input_keys = kwargs.get('inputs')
        self.keep_keys = self.target_keys+self.input_keys
        self.mapping_column_indices = self.get_column_idx_mapping()
        self.input_col_idxs = [self.mapping_column_indices[col_idx] for col_idx in self.input_keys]
        self.output_col_idxs = [self.mapping_column_indices[col_idx] for col_idx in self.target_keys]

    def get_column_idx_mapping(self):
        return {idx: col for col, idx in zip(self.data.columns, range(len(self.data.columns)))}
    
    def write_input_file(self):
        pass 