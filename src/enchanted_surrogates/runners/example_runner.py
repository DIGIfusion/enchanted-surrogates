import os
from .base_runner import Runner
from time import sleep

class ExampleRunner(Runner):
    
    def __init__(self, *args, **kwargs):
        self.parameter_mode = kwargs.get('parameter_mode', 0)

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        # Implementation for the example runner
        # Logic should follow something like
        # - via parser, write some input files there
        # - run the code
        # - return outputs
        #   - 'output' should be the most important output used in active learning methods,
        #     it should be a single float, integer or class
        #   - 'success' indicates if the run sucessfully created output or if it crashed etc
        #   - Any other code outputs of interest can be places in the returned dictionary. They
        #     will be collected and available in enchanted surrogates created datasets.
        #   - for the datasets to be created properly the return dictionary values should not be
        #     iterables, only base types such as int, float, string, boolean...

        outfile = os.path.join(run_dir, "output.txt")

        # TODO make example parser
        # parser.write_input(run_dir, params)
        if self.parameter_mode==0:
            c1 = params["c1"]
            c2 = params["c2"]
            os.system(f"python3 -c 'print({c1} + {c2})' >> {outfile}")
            
            with open(outfile, 'r') as f:
                output = float(f.read().strip())
            sleep(0.5)
            result = {"output_1": output, "success": True, 'other_code_output_A': 'something_from_code_A'}
        elif self.parameter_mode==1:
            c3 = params["c3"]
            c4 = params["c4"]
            os.system(f"python3 -c 'print({c3} + {c4})' >> {outfile}")
            with open(outfile, 'r') as f:
                output = float(f.read().strip())
            sleep(0.5)
            result = {"output_2": output, "success": True, 'other_code_output_B': 'something_from_code_B'}
        elif self.parameter_mode==2:
            c5 = params["c5"]
            c6 = params["c6"]
            os.system(f"python3 -c 'print({c5} + {c6})' >> {outfile}")
            with open(outfile, 'r') as f:
                output = float(f.read().strip())
            sleep(0.5)
            result = {"output_3": output, "success": True, 'other_code_output_C': 'something_from_code_C'}
        else:
            raise ValueError('THE SET PARAMETER MODE DOES NOT MATCH ANY KNOWN PARAMETER MODE FOR EXAMPLE RUNNER')
        # TODO execute some code actually
        # for now, just sum the two parameters

        # TODO read output
        # parser.read_output(run_dir)

        return result 
