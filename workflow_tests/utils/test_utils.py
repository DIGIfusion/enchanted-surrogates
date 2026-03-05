import os
from enchanted_surrogates.supervisor.supervisor import Supervisor

def get_run_dir_count(path: str):
    # TODO filter by folder names
    return len(next(os.walk(path))[1]) - 1

