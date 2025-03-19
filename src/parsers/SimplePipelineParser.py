from parsers import SIMPLEparser

simple_parser = SIMPLEparser()
from dask.distributed import print
def simple_out_to_simple_in(last_out_dir, next_run_dir, run_id, future):
    """
        Handles parsing the output from simple_out_dir to make a run_dir at simple_in_dir
        Args:
        Returns:
        Raises:
    """
    params_out = simple_parser.read_output_file(last_out_dir) 
    simple_parser.write_input_file(params_out, run_dir=next_run_dir)
    
    return last_out_dir, next_run_dir, run_id