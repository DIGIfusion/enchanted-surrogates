try:
    from base import Parser # add . for relative import
except:
    try:
        from .base import Parser # add . for relative import
    except:
        raise ImportError

import subprocess
import os
import numpy as np
import f90nml
from typing import List
from copy import deepcopy


class GENEparser(Parser):
    """An I/O parser for GENE

     Attributes
    ----------

    Methods
    -------
    write_input_file
        Writes the inputfile for a single set of parameters.

    read_output_file
        Reads the output file to python format

    """
    def __init__(self, base_params_dir):
        """
        Generates the base f90nml namelist from the GENE parameters file at base_params_dir.

        Parameters
        ----------
            base_params_dir (string or path): The directory pointing to the base GENE parameters file.
            The base GENE parameters file must contain all parameters necessary for GENE to run.
            Any parameters to be sampled will be inserted into the base parameter file before each run.
            Any value of a sampled parameter in the base file will be ignored. 
        Returns
        -------
            Nothing 
        """
        self.base_namelist = f90nml.read(base_params_dir) #odict_keys(['parallelization', 'box', 'in_out', 'general', 'geometry', '_grp_species_0', '_grp_species_1', 'units'])


class GENE_single_parser(GENEparser):

    def write_input_file(self, params: dict, run_dir):
        """
        Write the GENE input file to the run directory specified. 
        
        Parameters
        ----------
            params (dict): The keys store strings of the names of the parameters as specified in the enchanted surrogates *_config.yaml configuration file.
            The values stores floats of the parameter values to be ran in GENE.

            rprint('Writing to', run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'input.tglf')
            subprocess.run(['touch', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')un_dir (string or path): The file system directory where runs are to be stored

        """
        print('Writing to', run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'parameters')
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')

        params_keys = list(params.keys())
        params_values = list(params.values())
        patch = {}

        for key, value in zip(params_keys,params_values):
            group_name, variable_name = key.split('-')
            if list(patch.keys()).count(group_name) > 0:
                patch[group_name][variable_name] = value
            else: patch[group_name] = {variable_name:value}

        namelist = self.base_namelist
        patch = f90nml.namelist.Namelist(patch)
        namelist.patch(patch)
        
        f90nml.write(namelist, input_fpath)

    # what is returned here is returned to the runner for a single code run, which goes though the base executor to get to the future 
    def read_output_file(self, run_dir: str):
        raise NotImplementedError
    
class GENE_scan_parser(GENEparser): 
    def write_input_file(self, params: dict, run_dir, file_name='parameters'):
        namelist = self.base_namelist
        namelist_string=str(namelist)
        
        #populate params: dict with all omn's required. Since each should be identical
        for i in range(namelist_string.count('&species')):
            params[f'_grp_species_{i}-omn'] = params['species-omn']
        params.pop('species-omn')

        # checking run dir exists and making Path for scan file
        print('Writing to', run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, file_name)
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')
        
        def find_nth_occurrence(string, sub_string, n):
            start_index = string.find(sub_string)
            while start_index >= 0 and n > 1:
                start_index = string.find(sub_string, start_index + 1)
                n -= 1
            return start_index

        # finds the string location at the end of the line for a variable, just before \n
        def var_end_loc(namelist_string: str, param_key):
            group_name, var_name = param_key.split('-')
            # print('GROUP NAME',group_name,'VAR NAME',var_name)
            group_ordinal = 0 #0 is the 1st
            if len(group_name.split('_'))>1:
                # print('MORE THAN ONE GROUP OF SAME NAME')
                _, _, group_name, group_ordinal = group_name.split('_')
                group_ordinal = int(group_ordinal)+1

            # print('GROUP NAME',group_name,'VAR NAME',var_name, 'ORDIANL',group_ordinal)

            group_start = find_nth_occurrence(namelist_string, group_name, group_ordinal)
            group_end = group_start+namelist_string[group_start:].find(f'/')
            # print('GROUP',namelist_string[group_start:group_end])

            var_start = group_start+namelist_string[group_start:group_end].find(var_name)
            var_end = var_start+namelist_string[var_start:group_end].find("\n")
            # print('VARLOC',var_start,var_end)
            # print('VAR',namelist_string[var_start:var_end])
            # print('START',namelist_string[var_start],'END',namelist_string[var_end])
            return var_end

        # Making kymin scanlist
        kymin = params['box-kymin']
        scanlist = f'      !scanlist: {kymin[0]}'
        for ky in kymin[1:]:
            scanlist += f', {ky}'
        var_end = var_end_loc(namelist_string, 'box-kymin')
        namelist_string = namelist_string[:var_end] + scanlist + namelist_string[var_end:]
        #----------------------
        
        #determines the irdinal position of the scanned paramters for var_name
        def var_ordinal(namelist,var_name):
            namelist_string = str(namelist)
            kymin_loc = namelist_string.find(var_name)
            var_ordinal = namelist_string[:kymin_loc].count('!scan')
            return var_ordinal

        def make_scanwith(namelist, values):
            kymin_loc = var_ordinal(namelist, 'kymin')
            # print(f'Generating scanwith string for {var_name}')
            
            # print('VALUES',values)
            scanwith = f'       !scanwith: {kymin_loc}, {values[0]}'
            for v in values[1:]:
                scanwith += f', {v}'
            return scanwith

        scanwith = {}
        params.pop('box-kymin')
        for var_name in params.keys():
            scanwith[var_name] = make_scanwith(namelist, values=params[var_name])
        # print('SCANWITH',scanwith)
            # finds the string location at the end of the line for a variable, just before \n
        
        
        for param_key in list(params.keys()):
            var_end = var_end_loc(namelist_string,param_key)
            namelist_string = namelist_string[:var_end] + scanwith[param_key] + namelist_string[var_end:]
        # print(namelist_string)        
        
        #Writing the final namelist stirng to file. This is the scan parameters file.
        with open(input_fpath, 'w') as file:
            file.write(namelist_string)        

    def read_output_file(self, run_dir: str):
        raise NotImplementedError
    
if __name__ == '__main__':
    bounds = [[0.1, 300],[2,3.5],[4,6.8]]
    # params = {'box-kymin':100.1, '_grp_species_0-omt': 2.75, '_grp_species_1-omt':5.1}
    generator = np.random.default_rng(seed=238476592)
    omn = generator.uniform(5,60,5)
    params = {'box-kymin':generator.uniform(0.05,1,5),
          'species-omn':omn,
          '_grp_species_1-omt':generator.uniform(10,70,5)}
    parser = GENE_scan_parser(base_params_dir = os.path.join(os.getcwd(),'src','parsers','parameters_base'))
    parser.write_input_file(params,run_dir=os.getcwd(),file_name='parameters_scanwith')