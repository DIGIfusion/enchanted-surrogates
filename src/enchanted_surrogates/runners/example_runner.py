import os
from .base_runner import Runner


class ExampleRunner(Runner):

    def single_code_run(self, run_dir: str, params: dict = None) -> dict:
        # Implementation for the example runner
        # Logic should follow something like
        # - create the run directory
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

        return {"output": output, "success": True, 'other_code_output': 'something_from_code'}
