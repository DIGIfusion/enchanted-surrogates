import xarray as xr
import numpy as np
import re
import os

from TPED.__init__ import ureg


class GENEProfileReader:
    def __init__(self, input_filepath_list):
        if isinstance(input_filepath_list, str):
            filepath_list = self._find_profiles_in_dir(input_filepath_list)
        elif isinstance(input_filepath_list, list):
            filepath_list = input_filepath_list
        else:
            raise ValueError('Input filepath must be a string or list of strings.')
        self.filepath_list = filepath_list

    def write_GENEPfile(self, output_filepath, profile_xarray):

        if isinstance(profile_xarray, dict):
            profile_xarray = self.output_profile_xarray(profile_xarray)
        if not isinstance(profile_xarray, xr.Dataset):
            raise ValueError('Profile data must be a dictionary or xarray Dataset.')
        
        all_species_list = [profile_xarray[var].name[-1] for var in profile_xarray.data_vars]
        species_list = list(set(all_species_list))

        for spec_str in species_list:
            spec_xarray = profile_xarray[[var for var in profile_xarray.data_vars if var.endswith(spec_str)]]
            temp_str = f'T{spec_str}'
            dens_str = f'n{spec_str}'

            temp_units = spec_xarray[temp_str].attrs['original_units']
            dens_units = spec_xarray[dens_str].attrs['original_units']

            spec_filename = f'profiles_{spec_str}'
            profile_savepath = os.path.join(output_filepath, spec_filename)

            # Open file to write
            with open(profile_savepath, 'w') as file:
                # file.write(f"# 1.rho_tor 2.rho_pol 3.T{spec_str}(kev) 4.n{spec_str}(10^19m^-3) \n#\n")
                file.write(f"# 1.rho_tor 2.rho_pol 3.T{spec_str}({temp_units}) 4.n{spec_str}({dens_units}) \n#\n")
                

                if temp_str in profile_xarray.data_vars and dens_str in profile_xarray.data_vars:

                    rho_tor = spec_xarray['rho_tor'].values
                    rho_pol = spec_xarray['rho_pol'].values
                    temp_data = spec_xarray[temp_str].values
                    dens_data = spec_xarray[dens_str].values
                    
                    # Check if rho_pol is NaN and prepare data lines
                    for rt, rp, temp, dens in zip(rho_tor, rho_pol, temp_data, dens_data):
                        rt_val = 'nan' if np.isnan(rt) else f"{rt:.18e}"
                        rp_val = 'nan' if np.isnan(rp) else f"{rp:.18e}"
                        temp_val = 'nan' if np.isnan(temp) else f"{temp:.18e}"
                        dens_val = 'nan' if np.isnan(dens) else f"{dens:.18e}"

                        line = f"{rt_val} {rp_val} {temp_val} {dens_val}\n"
                        file.write(line)
                else:
                    missing_variables = [v for v in [temp_str, dens_str] if v not in profile_xarray.data_vars]
                    raise ValueError(f"Data for species {spec_str} not complete in the dataset. Missing: {', '.join(missing_variables)}")



    def output_profile_dict(self, species_list=None):
        merged_profile_dict = {}

        if not (species_list is None or isinstance(species_list, list)):
            raise ValueError('Species list must be a list or None.')
        if (species_list is not None) and (len(self.filepath_list) != len(species_list)):
            raise ValueError('Number of species input must match number of filepaths.')

        for filepath_ind, filepath in enumerate(self.filepath_list):
            species = species_list[filepath_ind] if isinstance(species_list,list) else None
            profile_dict = self._single_profile_dict(filepath, species=species)  
            merged_profile_dict.update(profile_dict)
        
        return merged_profile_dict


    def output_profile_xarray(self, species_list=None):
        profile_dict = self.output_profile_dict(species_list=species_list)

        # Creating a combined, sorted array of all unique rho_tor and rho_pol values
        all_rho_tor = np.unique(np.concatenate([var_dict['rho_tor'] for var_dict in profile_dict.values()]))
        all_rho_pol = np.unique(np.concatenate([var_dict['rho_pol'] for var_dict in profile_dict.values()]))
        all_rho_tor.sort()
        all_rho_pol.sort()

        data_vars = {}
        for var, var_dict in profile_dict.items():
            rho_tor_indices = np.searchsorted(all_rho_tor, var_dict['rho_tor'])
            rho_pol_indices = np.searchsorted(all_rho_pol, var_dict['rho_pol'])
    
           
            # Main variable data array
            data_vars[var] = xr.DataArray(
                data=var_dict['data'],
                dims=['index'],  # Use a generic dimension name, since each data point corresponds to a position in 'rho_tor' and 'rho_pol'
                coords={
                    'index': np.arange(len(var_dict['data'])),  # Positional index for each data point
                    'rho_tor': ('index', all_rho_tor[rho_tor_indices]),  # Attach 'rho_tor' values to the data
                    'rho_pol': ('index', all_rho_pol[rho_pol_indices])   # Attach 'rho_pol' values to the data
                },
                attrs={
                    'units': var_dict['units'],
                    'original_units': var_dict['original_units']
                }
            )
        
        # Creating the xarray Dataset from the data_vars dictionary
        profile_dataset = xr.Dataset(data_vars)
        
        return profile_dataset


    def _single_profile_dict(self, filepath, species=None):

        with open(filepath, 'r') as file:
            lines = file.readlines()

        col_header = lines[0].strip().split()[1:]  # Extract column headers
        if len(col_header) != 4:
            raise FileExistsError(f'File given does not contain 4 columns and cannot be parsed: \n {filepath}')
        
        col_header = [re.sub(r'\d+\.', '', col) for col in col_header]  # Remove numbers and periods
        data = []

        for line in lines[1:]:
            if not line.startswith('#'):
                columns = line.strip().split()
                data.append([float(x) if x != 'nan' else np.nan for x in columns])

        species = self._species_checks(species, filepath)

        data = np.array(data)
        profile_dict = {}
        for i, quantity in enumerate(col_header):
            quantity_name, units_name = self._split_quantity_name_units(quantity)

            if quantity_name.startswith('T'):
                quantity_name = f'T{species}'
            elif quantity_name.startswith('n'):
                quantity_name = f'n{species}'                    
            
            if not quantity_name.startswith('rho_'):
                profile_dict[quantity_name] = {
                    'data': data[:, i],
                    'rho_tor': data[:, 0],
                    'rho_pol': data[:, 1],
                    'units': ureg(units_name),
                    'original_units': units_name
                }
        
        return profile_dict


    def _split_quantity_name_units(self, quantity_units: list):
        pattern = r'([a-zA-Z]+)\((.*?)\)'
        match = re.match(pattern, quantity_units)
        if match:
            quantity_name = match.group(1)
            units_name = match.group(2).lower()
        else:
            quantity_name = quantity_units
            units_name = None

        return quantity_name, units_name


    def _species_checks(self, species, filepath):
        if species is None:
            species = filepath.split('_')[-1][0]  # Extract species from filename if not specified
        while species not in ['e', 'i', 'z']:
            species = input(f'Invalid species: "{species}". Please specify either "e", "i", or "z". Or press "q" to quit.')
            if species == 'q':
                raise UserQuitException("User has chosen to quit.")
            elif species in ['e', 'i', 'z']:
                return species
        
        return species

    def _find_profiles_in_dir(self, input_filepath):
        profile_list = []

        if os.path.isdir(input_filepath):

            for filename in os.listdir(input_filepath):
                test_filepath = os.path.join(input_filepath, filename)
                if ('profiles_' in filename) and os.path.isfile(test_filepath):
                    profile_list.append(test_filepath)

            if len(profile_list) == 0:
                raise FileNotFoundError(f'No profile files found in directory: {input_filepath}')
        else:
            profile_list = input_filepath

        return profile_list



class UserQuitException(Exception):
    pass