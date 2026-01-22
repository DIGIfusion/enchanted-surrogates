if __name__ == "__main__":
    import sys
    import os
    import yaml
    import pickle
    import argparse
    import pysgpp
    from enchanted_surrogates.utils.get_batch_dirs import get_batch_dirs
    from enchanted_surrogates.utils.load_configuration import load_configuration
    from enchanted_surrogates.samplers.sgpp_sampler import SgppSampler
    
    # You must add the test_data_csv to the sampler args in the config file within the base_run_dir before running this script

    _, base_run_dir = sys.argv
    
    listdir = os.listdir(base_run_dir)
    config_file_name = [name for name in listdir if '.yaml' in name]
    if len(config_file_name) > 1:
        raise FileNotFoundError('More than one .yaml file in base_run_dir, not sure which to use as config file')
    config_file_name = config_file_name[0]
    print('CONFIG FOUND:', os.path.join(base_run_dir, config_file_name))
    config = load_configuration(os.path.join(base_run_dir, config_file_name))

    sampler_config = config.executor['sampler_config']
    sampler_config['base_run_dir'] = base_run_dir
    sgpp = SgppSampler(**sampler_config)
    sgpp.add_rmse_column_to_batch_info()