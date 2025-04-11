import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','src'))
import run 

def test_helena_pedestal_sample_betaN_iteration():
    config_path = os.path.join(os.path.dirname(__file__), 'configs','helena_pedestal_sample_betaN_iteration.yaml')
    
    args = run.load_configuration(config_path)
    os.system(f'rm -r {args['executor']['base_run_dir']}/*')
    run.main(args)
    
    