import os
import xarray as xr
import numpy as np

from TPED.projects.GENE_sim_reader.src.GENE_parameters_data import GeneParameters as GP
from projects.GENE_sim_reader.archive.ARCHIVE_GENE_field_data import GeneField as GF
from TPED.projects.GENE_sim_reader.src.GENE_nrg_data import GeneNrg as GN
from TPED.projects.GENE_sim_reader.src.GENE_omega_data import GeneOmega as GO

from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC


GENE_FILETYPES = ['parameters', 'omega', 'field', 'nrg']
# Expected keys set
INPUT_KEYS = {'time_criteria', 'nrg_quantities', 'species', 'fields'}


class GeneSimulation():

    def __init__(self, gene_file_filepath: str):
        self.directory = os.path.dirname(gene_file_filepath)
        self.suffix = GFC(gene_file_filepath).suffix_from_filepath()

        self.parameters_filepath = GFC(gene_file_filepath).switch_suffix_file('parameters')
        self.omega_filepath = GFC(gene_file_filepath).switch_suffix_file('omega')
        self.field_filepath = GFC(gene_file_filepath).switch_suffix_file('field')
        self.nrg_filepath = GFC(gene_file_filepath).switch_suffix_file('nrg')
        self.mom_filepath = GFC(gene_file_filepath).switch_suffix_file('mom')
        self.vsp_filepath = GFC(gene_file_filepath).switch_suffix_file('vsp')
        
        self.parameters_xarray = GP(self.parameters_filepath).parameters_filepath_to_xarray()
    
    def simulation_to_xarray(self, filetypes_list: list = [], input_quantities: dict = {}):

        simulation_xarray = self.parameters_xarray
        self._validate_input_keys(input_quantities, INPUT_KEYS)
        filetypes_list = [filetypes_list] if isinstance(filetypes_list, str) else filetypes_list

        for filetype in filetypes_list:
            if filetype not in GENE_FILETYPES:
                raise ValueError(f'Invalid filetype: {filetype}. Valid filetypes are: {GENE_FILETYPES}')

            if filetype == 'omega':
                add_xarray = GO(self.omega_filepath).omega_filepath_to_xarray()

            elif filetype == 'nrg':
                time_criteria = input_quantities.get('time_criteria', 'last')
                nrg_quantities = input_quantities.get('nrg_quantities', 'all')
                input_species = input_quantities.get('species', 'all')
                time_sampler_method = input_quantities.get('time_sampler_method', 'absolute')

                add_xarray = GN(self.nrg_filepath).nrg_filepath_to_xarray(time_criteria=time_criteria, 
                                                                          input_nrg_species=input_species, 
                                                                          input_nrg_quantities=nrg_quantities,
                                                                          time_sampler_method=time_sampler_method)
                
            elif filetype == 'field':
                time_criteria = input_quantities.get('time_criteria', 'last')
                input_fields = input_quantities.get('fields', 'all')
                time_sampler_method = input_quantities.get('time_sampler_method', 'absolute')

                add_xarray = GF(self.field_filepath).field_filepath_to_xarray(time_criteria=time_criteria, 
                                                                               input_fields=input_fields,
                                                                               time_sampler_method=time_sampler_method)

            simulation_xarray = self._merge_with_tolerance(simulation_xarray, add_xarray)
            
        return simulation_xarray



    def _merge_with_tolerance(self, main_da, add_da, tolerance=1e-5):
        # Adjusting coordinates within tolerance and ensuring all coordinates are considered
        aligned_coords = {coord: main_da.coords[coord] for coord in main_da.coords}

        # Update aligned_coords with add_da coordinates, checking for close matches
        for coord in add_da.coords:
            if coord in main_da.coords:
                main_values = main_da.coords[coord].values
                add_values = add_da.coords[coord].values
                new_values = []

                # Find values in add_values close to any in main_values or add them if no match
                for val in add_values:
                    diffs = np.abs(main_values - val)
                    min_diff = diffs.min()
                    if min_diff <= tolerance:
                        closest_val = main_values[diffs.argmin()]
                        new_values.append(closest_val)
                    else:
                        new_values.append(val)

                aligned_coords[coord] = np.unique(np.concatenate([main_da.coords[coord], new_values]))
            else:
                aligned_coords[coord] = add_da.coords[coord]

        # Assign adjusted coordinates to new datasets and combine them
        main_da_expanded = main_da.reindex({k: v for k, v in aligned_coords.items()}, fill_value=np.nan)
        add_da_expanded = add_da.reindex({k: v for k, v in aligned_coords.items()}, fill_value=np.nan)

        # Merge the expanded datasets
        return xr.merge([main_da_expanded, add_da_expanded])


    def _validate_input_keys(self, input_dict, expected_keys):
        """ Check if the input keys match the expected keys and warn if there are discrepancies. """
        input_keys = set(input_dict.keys())
        extra_keys = input_keys - expected_keys
        
        if extra_keys:
            raise ValueError(f"Warning: Some keys are not recognized: {extra_keys}\nExpected keys are: {expected_keys}")
    
    

