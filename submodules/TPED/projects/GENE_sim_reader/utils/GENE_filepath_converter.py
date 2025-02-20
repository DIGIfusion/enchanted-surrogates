import os
import re



VALID_FILETYPES = ["parameters", "omega", "nrg", "mom", "field"]

class GeneFilepathConverter:
    def __init__(self, filepath:str):
        self.filepath = filepath
        self.directory = os.path.dirname(self.filepath)
        self.filename = os.path.basename(self.filepath)
        self.validate_filepath()

    def validate_filepath(self, filepath:str=None):
        filepath = filepath if filepath else self.filepath
        filename = os.path.basename(filepath)

        if not any(filename.startswith(filetype) for filetype in VALID_FILETYPES):
            raise ValueError(f"The file name {filename} does not start with any of the valid filetypes: {VALID_FILETYPES}.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at {filepath} does not exist.")
        if not os.path.isfile(filepath):
            raise ValueError(f"The path {filepath} is not a file.")
        
    ####################################################################################################
    #################### Functions to switch to different filetypes ####################################
    ####################################################################################################

    def suffix_from_filepath(self):
        match = re.search(r'\d+$', self.filename)
        suffix = match.group() if match else '.dat'
        if suffix != '.dat' and len(suffix) != 4:
            raise ValueError(f"Ensure {self.filename} has four digits like '0002'")
        return suffix
    
    def switch_suffix_file(self, filetype:str):
        suffix = self.suffix_from_filepath()
        
        # If suffix has a '.dat', no change is needed; otherwise, prepend an underscore
        mod_suffix = suffix if '.dat' in suffix else '_' + suffix
        new_filepath = os.path.join(self.directory, filetype + mod_suffix)
        self.validate_filepath(new_filepath)
        
        return new_filepath




# from TPED.projects.GENE_sim_reader.src.dict_parameters_data import parameters_filepath_to_dict
# from TPED.projects.GENE_sim_reader.src.dict_omega_data import omega_filepath_to_dict
# from TPED.projects.GENE_sim_reader.src.dict_nrg_data import nrg_filepath_to_dict



# class GENEFilepathToDict:
#     def __init__(self, filepath:str):
#         self.filepath = filepath
#         self.directory = os.path.dirname(self.filepath)
#         self.filename = os.path.basename(self.filepath)
#         self.validate_filepath()
        
#     def validate_filepath(self):
#         if not os.path.exists(self.filepath):
#             raise FileNotFoundError(f"The file at {self.filepath} does not exist.")
#         if not os.path.isfile(self.filepath):
#             raise ValueError(f"The path {self.filepath} is not a file.")
#         if not any(os.path.basename(self.filepath).startswith(filetype) for filetype in valid_filetypes):
#             raise ValueError(f"The file name {os.path.basename(self.filepath)} does not start with any of the valid filetypes: {valid_filetypes}.")

#     def filepath_to_dict(self):
#         try:
#             if "parameters" in self.filepath:
#                 return parameters_filepath_to_dict(self.filepath)
#             elif "omega" in self.filepath:
#                 return omega_filepath_to_dict(self.filepath)
#             elif "nrg" in self.filepath:
#                 return nrg_filepath_to_dict(self.filepath)
#             else:
#                 raise ValueError("Unsupported file type or content.")
#         except Exception as e:
#             raise RuntimeError(f"Failed to convert file to dictionary: {e}")






