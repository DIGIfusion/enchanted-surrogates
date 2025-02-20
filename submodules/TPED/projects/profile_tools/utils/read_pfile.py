import xarray as xr
import numpy as np
import re


from TPED.__init__ import ureg



class PFileReader:

    def __init__(self, pfile_filepath):
        self.pfile_filepath = pfile_filepath
    
    def output_profile_dict(self):

        column_names, data_list = self._parse_pfile_data()
        quantity_dict = {}

        for col_ind, col_list in enumerate(column_names):
            data_chunk = data_list[col_ind]

            new_array = np.array(data_chunk, dtype=float)
            
            psinorm = new_array[:, 0]
            quantity = new_array[:, 1]
            gradient = new_array[:, 2]

            
            quantity_name, units_name = self._split_quantity_name_units(col_list)
            grad_unit = col_list[2]

            quantity_dict[quantity_name] = {
                'data': quantity,
                'units': units_name,
                'psinorm': psinorm,
                'gradient': gradient,
                'grad_units': grad_unit
            }

        return quantity_dict





    def output_profile_xarray(self):

        profile_dict = self.output_profile_dict()

        # Creating a combined, sorted array of all unique psinorm values
        all_psinorm = np.unique(np.concatenate([var_dict['psinorm'] for var_dict in profile_dict.values()]))
        all_psinorm.sort()

        data_vars = {}

        for var, var_dict in profile_dict.items():
            
            # Continue with the original data processing logic (excluding the averaging or aggregation)
            var = re.sub(r'^(te|ti|tz)$', lambda m: m.group(0).title(), var)

            # Main variable data array
            data_vars[var] = xr.DataArray(
                data=var_dict['data'],
                dims=['psinorm'],
                coords={'psinorm': var_dict['psinorm']},  # Using the original psinorm values
                attrs={
                    'units': ureg(var_dict['units']),
                    'original_units': var_dict['units']  # Store the original units for future reference
                }
            )
            
            # Gradient data array, use the original gradient values
            data_vars[f'{var}_gradient'] = xr.DataArray(
                data=var_dict['gradient'],
                dims=['psinorm'],
                coords={'psinorm': var_dict['psinorm']},  # Use the original psinorm values
                attrs={
                    'original_units': var_dict['grad_units']
                }
            )

        # Creating the dataset
        profile_dataset = xr.Dataset(data_vars)

        return profile_dataset


        
    

    def _parse_pfile_data(self):
        with open(self.pfile_filepath, 'r') as file:
            lines = file.readlines()
        
        column_names = []
        data = []
        data_per_column = []
        collect_data = False

        # Pattern to match lines that consist of three floating-point numbers
        numeric_line_pattern = re.compile(r'^\s*-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s*$')

        for line in lines:
            line = line.strip()  # Strip any leading or trailing whitespace

            if "ION SPECIES" in line:
                collect_data = False
                continue
            
            if 'psinorm' in line:
                collect_data = True  # Start collecting data
                pattern = r'\s+(?![^\(]*\))'
                column_headers = re.split(pattern, line)
                cleaned_column_headers = column_headers[1:]  # Remove '256'
                
                if len(cleaned_column_headers) != 3:
                    raise ValueError(f"Expected 3 columns, but got: {cleaned_column_headers}")
                column_names.append(cleaned_column_headers)

                if data_per_column:
                    data.append(data_per_column)
                    data_per_column = []

                continue  # Skip further processing of this line since it’s a header

            elif collect_data and numeric_line_pattern.match(line):
                
                parts = line.split()
                if len(parts) == 3:  # Ensure data matches the number of columns
                    # print(parts)
                    data_per_column.append([float(part) for part in parts])
                

        # Append the last block of data if any remains
        if data_per_column:
            data.append(data_per_column)

        return column_names, data

    def _split_quantity_name_units(self, col_name_list: list):
        quantity_units = col_name_list[1]        
        pattern = r'([a-zA-Z]+)\((.*?)\)'
        match = re.match(pattern, quantity_units)
        if match:
            quantity_name = match.group(1)
            units_name = match.group(2).lower()

            if units_name == 'no data':
                units_name = None
            # print(quantity_name, units_name)
        else:
            quantity_name = quantity_units
            units_name = None

        return quantity_name, units_name
