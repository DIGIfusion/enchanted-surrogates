import uuid
import os

def make_run_dir(base_run_dir:str, prepend:str):
    return os.path.join(base_run_dir, prepend+'-'+str(uuid.uuid4())) # TODO. uuid.uuid should probably have a random seed ? 