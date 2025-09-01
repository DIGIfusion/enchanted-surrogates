from .base_runner import Runner

import os

class ExampleRunner(Runner):

    def single_code_run(self, run_dir: str, params: dict = None):
        # Implementation for the example runner

        # Logic should follow something like 
        # --- create the run directory 
        # --- via parser, write some input files there 
        # --- run the code 
        # --- return  outputs

        os.makedirs(run_dir, exist_ok=True)


        # TODO make example parser 
        # parser.write_input(run_dir, params)
        c1 = params["c1"]
        c2 = params["c2"]
        outfile = os.path.join(run_dir, "output.txt")

        # TODO execute some code actually 
        # for now, just sum the two parameters 
        
        os.system(f"python3 -c 'print({c1} + {c2})' >> {outfile}")

        # TODO read output 
        # parser.read_output(run_dir) 

        with open(outfile, 'r') as f:
            output = float(f.read().strip())
            
        return output
