import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','src'))
import run 
import uuid
import numpy as np
from parsers.HELENAparser import HELENAparser

def is_valid_uuid(val):
    try:
        uuid_obj = uuid.UUID(val, version=4)
    except ValueError:
        return False      
    return str(uuid_obj) == val

def test_helena_pedestal_sample_betaN_iteration():
    config_path = os.path.join(os.path.dirname(__file__), 'configs','helena_pedestal_sample_betaN_iteration.yaml')
    args = run.load_configuration(config_path)
    base_run_dir = args.executor['base_run_dir']
    os.system(f"rm -r {args.executor['base_run_dir']}/*")    
    run.main(args)
    dirs = os.listdir(base_run_dir)
    run_dirs = [d for d in dirs if is_valid_uuid(d)]
    
    
    beta_tolerance = args.runner['other_params']['beta_tolerance']
    beta_target = args.runner['other_params']['beta_N_target']
    parser = HELENAparser()
    
    for run_dir in run_dirs:
        success1, mercier_stable, ballooning_stable = parser.read_output_file(run_dir)
        fort20 = os.path.join(run_dir, 'fort.20')
        with open(fort20, 'r') as file:
            for line in file:
                if 'NORM. BETA' in line:
                    break
            beta_N = float(line.strip().split(' ')[-1])*1e2
        success2 = np.abs(beta_N-beta_target) < beta_tolerance
        success = success1 and success2
        assert success
    
    