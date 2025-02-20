import struct
import os
import numpy as np
import xarray as xr
from os.path import getsize

from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC
from TPED.projects.GENE_sim_reader.src.GENE_parameters_data import GeneParameters as GP

# List of field quantities and their corresponding indices
FIELD_QUANT_IND = [('field_phi', 0), ('field_apar', 1), ('field_bpar', 2)]

class GeneField:
    def __init__(self, field_filepath: str):
        self.field_filepath = field_filepath
        # Convert field filepath to parameter filepath and load parameters
        param_filepath = GFC(self.field_filepath).switch_suffix_file('parameters')
        self.param_dict = GP(param_filepath).parameters_filepath_to_dict()
        self._calc_file_specs()
    
    def field_filepath_to_xarray(self, time_criteria: str = 'last', input_fields: str = 'all'):
        field_dict = self.field_filepath_to_dict(time_criteria, input_fields)
        return self.field_dict_to_xarray(field_dict)

    def field_dict_to_xarray(self, field_dict: dict):
        """
        Convert field dictionary to xarray dataset.
        
        Args:
        - field_dict (dict): A dictionary containing field data and relevant information.
        
        Returns:
        - xarray.Dataset: An xarray dataset containing field data.
        """
        filepath = field_dict.pop('filepath', None)
        directory = os.path.dirname(filepath)
        suffix = GFC(filepath).suffix_from_filepath()

        # Prepare initial field data and coordinates
        field_phi_data = np.array(field_dict['field_phi'])
        time_data = np.array(field_dict['time'])
        zgrid_data = np.array(field_dict['zgrid'])

        # Determine the minimum time length across all fields
        min_length = len(time_data)
        fields_to_add = ['field_phi']  # already included
        for field in ['field_apar', 'field_bpar']:
            if field in field_dict and field_dict[field]:
                field_data = np.array(field_dict[field])
                min_length = min(min_length, len(field_data))
                fields_to_add.append(field)

        # Create a new dataset with trimmed data
        field_dataset = xr.Dataset(
            coords={
                'time': time_data[:min_length],
                'zgrid': zgrid_data
            }
        )

        # Add each field to the dataset
        for field in fields_to_add:
            field_data = np.array(field_dict[field]) if field in field_dict else field_phi_data
            field_dataset[field] = (['time', 'zgrid'], field_data[:min_length])

        field_dataset.attrs['filepath'] = filepath
        field_dataset.attrs['directory'] = directory
        field_dataset.attrs['suffix'] = suffix

        return field_dataset
        

    
    def field_filepath_to_dict(
        self, 
        time_criteria: str = 'last', 
        input_fields: str = 'all'
    ) -> dict:
        """
        Reads field data from a binary file and returns a dictionary containing relevant information.

        Args:
        - time_criteria (str): Specifies the time instance(s) to extract. Options:
            - 'last': Extracts data at the last available time.
            - 'first': Extracts data at the first available time.
        - input_fields (str): Specifies the field column(s) to extract. Options:
            - 'all': Extracts all fields.
            - str: Extracts the specified field column (phi OR apar OR bpar).
            - list: Extracts multiple specified field columns (i.e. ['phi', 'bpar']).

        Returns:
        - dict: A dictionary containing field data and relevant information.
        """

        # Initializing the field dictionary with default values
        field_dict = {'filepath': self.field_filepath, 'time': [],
                      'field_phi': [], 'field_apar': [], 
                      'field_bpar': [], 'zgrid': []}

        # Sample field names based on input_fields parameter
        sampled_field_names = self._sample_name_ind(FIELD_QUANT_IND, input_fields)

        # Reading binary field data
        with open(self.field_filepath, 'rb') as file:
            # Extracting time values from the field file
            all_time_values = self._retrieve_time_values(file)

            # Handling different time extraction options
            if time_criteria == 'last':
                time_index_list = [all_time_values.index(max(all_time_values))]
            elif time_criteria == 'first':
                time_index_list = [all_time_values.index(min(all_time_values))]
            elif time_criteria == 'all':
                time_index_list = list(range(len(all_time_values)))
            else:
                raise ValueError(f'Ensure choose time is given as "last" or "first"')

            # Add time values based on chosen time value(s)
            field_dict['time'] = [all_time_values[time_ind] for time_ind in time_index_list]

            # If no time values are found, return empty dictionary
            if field_dict['time'] == []:
                return field_dict

            # Cycle through time values
            for time_index in time_index_list:
                # Cycle through field names from sampled_field_names
                for field_name, field_ind in sampled_field_names:
                    # Calculate offset in field file to retrieve data
                    offset = (time_index * self.time_offset) + (field_ind * self.field_offset) + self.offset_bias
                    file.seek(offset)

                    try:
                        flat_data_array, zgrid = self._data_array_flattened(file)
                    except:
                        return field_dict
                    
                    # Appending field data into field dictionary
                    field_dict.setdefault(field_name, []).append(flat_data_array)

                    # Check if 'zgrid' key exists and if its list is empty before appending
                    if 'zgrid' not in field_dict or len(field_dict['zgrid']) == 0:
                        field_dict['zgrid'] = zgrid
                    
        return field_dict



    def _sample_name_ind(self, full_name_ind_list: list, input_names: str) -> list:
        """Return sampled quantities & indices from a full list.
        If input_names is 'all', return all quantities & indices.
        Raises ValueError if any name in input_names is not found in full_name_ind_list.
        """
        if input_names == 'all':
            return full_name_ind_list

        # Split the input_names into a list if it's a string of names
        input_name_list = input_names.split() if isinstance(input_names, str) else input_names

        # Filter full_name_ind_list to include only items whose names are in input_name_list
        filtered_list = [(name, ind) for name, ind in full_name_ind_list if name in input_name_list]

        # Check if all input_names are found in full_name_ind_list
        if len(filtered_list) != len(input_name_list):
            missing_names = set(input_name_list) - {name for name, ind in filtered_list}
            available_names = ', '.join(name for name, _ in full_name_ind_list)
            raise ValueError(f"Quantity not found: '{', '.join(missing_names)}'.\n Please chose from the following:\n {[available_names]}")

        return filtered_list



    def _retrieve_time_values(self, file):
        """
        Retrieve all time values from the binary field file.
        
        Args:
        - file: The binary file object.

        Returns:
        - list: A list of time values.
        """
        all_time_values = []
        timeentry_size = self.timeentry.size

        for _ in range(int(getsize(self.field_filepath) / (self.leapfld + timeentry_size))):
            time = float(self.timeentry.unpack(file.read(timeentry_size))[1])
            file.seek(self.leapfld, 1)
            all_time_values.append(time)
        
        return all_time_values

    def _calc_file_specs(self):
        """
        Calculate specifications for reading the field file based on the parameter dictionary.
        """
        param_dict = self.param_dict
        
        # Extracting parameters from the parameter dictionary
        nx, ny, nz, n_fields, precision, endianness = (
            param_dict['nx0'],
            param_dict['nky0'],
            param_dict['nz0'],
            param_dict['n_fields'],
            param_dict['PRECISION'],
            param_dict['ENDIANNESS']
        )

        # Setting sizes based on precision
        self.intsize = 4
        realsize = 8 if precision == 'DOUBLE' else 4
        complexsize = 2 * realsize
        entrysize = nx * ny * nz * complexsize
        self.leapfld = n_fields * (entrysize + 2 * self.intsize)

        # Creating NumPy dtype for complex numbers based on precision
        self.complex_dtype = np.dtype(np.complex64) if precision == 'SINGLE' else np.dtype(np.complex128)

        # Setting the format string based on endianness and precision
        format_string = '>' if endianness == 'BIG' else '='
        format_string += 'ifi' if precision == 'SINGLE' else 'idi'
        self.timeentry = struct.Struct(format_string)
        self.timeentry_size = self.timeentry.size

        # Calculating offsets and biases for reading field data
        self.time_offset = (self.timeentry_size + self.leapfld)
        self.field_offset = (entrysize + 2 * self.intsize)
        self.offset_bias = self.timeentry_size + self.intsize

    def _data_array_flattened(self, file):
        """
        Read and flatten data array from the binary field file.
        
        Args:
        - file: The binary file object.

        Returns:
        - tuple: A tuple containing the flattened data array and the zgrid array.
        """
        nx, ny, nz = self.param_dict['nx0'], self.param_dict['nky0'], self.param_dict['nz0']

        # Reading and reshaping data array
        data_array = np.fromfile(file, count=nx * ny * nz, dtype=self.complex_dtype).reshape(nz, ny, nx)

        # Calculate zgrid values
        dz = float(2.0) / float(nz)
        ntot = nz * nx
        zgrid = np.arange(ntot) / float(ntot - 1) * (2 * nx - dz) - nx

        # Initialize flattened array
        flattened_array = np.zeros(ntot, dtype='complex128')
        half_nx_int = int(nx / 2)

        for i in range(half_nx_int):
            # Flatten positive and negative frequency components
            lower_end = (i + half_nx_int) * nz
            upper_end = (i + half_nx_int + 1) * nz
            flattened_array[lower_end:upper_end] = data_array[:, 0, i]

            lower_end_neg = (half_nx_int - i - 1) * nz
            upper_end_neg = (half_nx_int - i) * nz
            flattened_array[lower_end_neg:upper_end_neg] = data_array[:, 0, -1 - i]

        return flattened_array, zgrid
