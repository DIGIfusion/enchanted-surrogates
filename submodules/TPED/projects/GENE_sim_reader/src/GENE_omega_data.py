#!/usr/bin/env python3

import os
import xarray as xr

from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC

####################################################################################################
########################## Convert omega filepath to dict #####################################
####################################################################################################

class GeneOmega:

    def __init__(self, omega_filepath:str):
        self.omega_filepath = omega_filepath
        # self.omega_dict = self.omega_filepath_to_dict(self.omega_filepath)

    def omega_filepath_to_xarray(self):
        omega_dict = self.omega_filepath_to_dict()
        return self.omega_dict_to_xarray(omega_dict)

    def omega_dict_to_xarray(self, omega_dict: dict):

        # Extract 'filepath' from the dictionary
        filepath = omega_dict.pop('filepath', None)
        directory = os.path.dirname(filepath)
        suffix = GFC(filepath).suffix_from_filepath()

        # Convert OrderedDict to xarray dataset
        omega_dataset = xr.Dataset(omega_dict)

        # Add filepath as an attribute
        omega_dataset.attrs['filepath'] = filepath
        omega_dataset.attrs['directory'] = directory
        omega_dataset.attrs['suffix'] = suffix

        return omega_dataset

    def omega_filepath_to_dict(self):
        check_empty = (os.stat(self.omega_filepath).st_size == 0)

        if check_empty:
            return {}
        else:
            omega_dict = {'filepath': self.omega_filepath}

        if not check_empty:
            # Read in the omega file
            with open(self.omega_filepath, 'r') as file:
                lines = file.readlines()

            # Split lines and put values into omega dict
            for line in lines:
                items = line.split()

                # Add the filename, filepath, and suffix to the omega dictionary
                omega_dict['kymin'] = float(items[0])
                omega_dict['gamma'] = float(items[1])
                omega_dict['omega'] = float(items[2])   

        return omega_dict

