"""
# executors/DaskExecutor.py

Contains logic for executing surrogate workflow on Dask.

"""

import time
from common import S
from .base import Executor, run_simulation_task
import os


class SlurmExecutor(Executor):
    """
    Handles execution of surrogate workflow with slurm arrays
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_run_dir: str = kwargs.get("base_run_dir", None)
        self.sbatch: list = kwargs.get("sbatch", None)
        self.num_paralell_jobs: int = kwargs.get("num_paralell_jobs", None)
        self.config_path: str = kwargs.get("config_path", None)
    def clean(self):
        """
        """
        NotImplementedError()

    def submit_batch_of_params(self, param_list: list[dict]) -> list:
        """Submits a batch of parameters to the class

        Raises:
            ValueError: If the configuration is incomplete for ACTIVE sampler types.
        """
        # Set-up run directory
        all_params_string = ''
        all_dir_string = ''
        for params in param_list:
            run_dir = ''.join([f'_{k}-{v}_' for k, v in params.items()])
            run_dir = os.system(f'mkdir {os.path.join(self.base_run_dir, run_dir)} -p')
            all_dir_string = all_dir_string + run_dir + ' '
            all_params_string = all_params_string + "\'"+str(params)+"\'" + " "
            # run_simulation_task(self.runner_args, params, self.base_run_dir)

        #Make master slurm script
        num_samples = len(params)
        this_script_path = os.path.abspath(__file__)
        sbatch_array = [f'#SBATCH --array=1-{num_samples}%{self.num_paralell_jobs}',
                         '## LIST OF PARAMETERS AS JSON STRINGS',
                        f'PARAMS = ({all_params_string})',
                        f'DIRS = ({all_dir_string})',
                         '# Get the parameters for this job',
                         'PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}',
                         '# Get the run_dir for this job',
                         'DIR=${DIRS[$SLURM_ARRAY_TASK_ID]}',
                         '# Run the code',
                        f'python3 {this_script_path} $PARAM $DIR \'{self.config_path}\''
                        ]
        sbatch_str = '\n'.join(self.sbatch + sbatch_array)
        os.system('touch slurm_simple.sh')
        with open('slurm_simple.sh', 'w') as file:
            file.write(sbatch_str)

        
    def start_runs(self):
        sampler_interface: S = self.sampler.sampler_interface
        print(100 * "=")
        print("Creating initial runs")

        print("Generating samples:")
        initial_parameters = self.sampler.get_initial_parameters()
        print("Starting Slurm Arrays")

        if sampler_interface in [S.SEQUENTIAL]:
            print("Sampler is Suquential and Not Active")
            self.submit_batch_of_params(initial_parameters)

        elif sampler_interface in [S.BATCH]:
            NotImplementedError()
        elif sampler_interface in [S.ACTIVE, S.ACTIVEDB]:
            NotImplementedError()

if __name__ == '__main__':
    # Executed by master slurm script
    import runners
    import json
    import sys
    import run
    params_from_sampler = json.loads(sys.argv[1])
    run_dir = sys.argv[2]
    config_filepath = sys.argv[3]
    args = run.load_configuration(config_filepath)
    runner_args = args.runner    
    runner = getattr(runners, runner_args["type"])(**runner_args)
    runner.single_code_run(params_from_sampler, run_dir)
    