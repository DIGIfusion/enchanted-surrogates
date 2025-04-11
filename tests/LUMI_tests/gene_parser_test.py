import os, sys
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src/')
if not src_path in sys.executable: 
    sys.path.append(src_path)
    
from parsers.GENEparser import GENEparser
from samplers.SobolSequence import SobolSequence
import f90nml
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src/'))
def test_gene_parser():
    os.system(f"rm -r {os.path.join(os.path.dirname(__file__),'files','tmp')}/*")
    sampler_args = {'type':'SobelSequence',
                    'bounds':[[0,1], [0,1], [0,1], [0,1], [0,1]],
                    'num_samples':10,
                    'parameters':['omt1','omt2','omn','kymin','coll']
    }
    sampler = SobolSequence(**sampler_args)            
    parser = GENEparser()
    samples = sampler.get_initial_parameters()
    base_params_file_path = os.path.join(os.path.dirname(__file__),'files','parameters_base_uq_highprec')
    for i, params in enumerate(samples):
        run_dir = os.path.join(os.path.dirname(__file__),'files','tmp',f'test_run_dir_{i}')
        os.makedirs(run_dir)
        parser.write_input_file(params, run_dir, base_params_file_path)
        namelist = f90nml.read(os.path.join(run_dir,'parameters'))
        # perform check
        for p in sampler.parameters:
            if p=='omn':
                p='omn1'
            sampled_value = params[p]
            group, var = parser.parameter_nml_map[p]
            file_value = namelist[group][var]
            assert sampled_value == file_value
    os.system(f"rm -r {os.path.join(os.path.dirname(__file__),'files','tmp')}/*")

if __name__ == '__main__':
    test_gene_parser()