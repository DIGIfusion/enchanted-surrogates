import os



class GeneFileFinder:

    def __init__(self, input_filepath:str):
        self.input_filepath = input_filepath



    def find_files(self, filetype:str='parameters', max_depth:int=3, smart_param_append:bool=True):
        """
        Finds files of a specific type buried within a directory structure up to a specified depth. Ignores searching inside directories with prefixes 'X_' and 'in_par'.

        Parameters:
        - input_filepath (str): The base filepath or a comma-separated string of filepaths to search.
        - filetype (str): The file type to search for.
        - max_depth (int, optional): The maximum depth to search within the directory structure (default is 3).
        - smart_param_append (bool, optional): If True, appends '_' to the filetype if multiple parameters files are found (default is True).

        Returns:
        - List[str]: A list of filepaths for the found files of the specified type (i.e. a list of parametes filepaths for a given directory).

        Raises:
        - FileNotFoundError: If the specified directory does not exist or if no matching files are found.
        - TypeError: If the input_filepath is not a string.
        """
        
        # Initialize an empty list to store the filepaths of the 'parameters/omega/nrg' files
        filetype_files = []

        # Convert the input_filepath to a list if it's a comma-separated string
        input_filepath_list = self.input_filepath if isinstance(self.input_filepath, list) else self.input_filepath.split(',')


        for path in input_filepath_list:

            # Check if the directory exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"The directory does not exist: {path}")
            # Check if the path is a string
            if not isinstance(path, str):
                raise TypeError(f"Filepath must be a string: {path}")

            # Check if the base filepath is a 'parameters/omega/nrg' file and append it to the list
            if filetype in os.path.basename(path):
                filetype_files.append(path)


            # Walk through the directory structure
            for root, dirs, files in os.walk(path):
                # Exclude directories with specified prefixes
                dirs[:] = [d for d in dirs if not any(d.startswith(prefix) for prefix in ['X_', 'in_par'])]

                # Calculate the depth of the current directory from the starting directory
                current_depth = root[len(path):].count(os.sep)

                # Check if the current depth is within the specified limit
                if current_depth <= max_depth:
                    for file in files:
                        
                        #change filetype to be added (parameters vs parameters_) if multiple filetypes are found
                        #this is only necessary for parameters files which can be input "parameters" or output "parameters_"
                        if (self._count_files_in_dir(root, filetype) > 1) and smart_param_append:
                            modifier = '_'
                        else:
                            modifier = ''
                        
                        # Check if the file starts with the specified filetype
                        if file.startswith(filetype + modifier):
                            filetype_path = os.path.join(root, file)
                            filetype_files.append(filetype_path)

        # Raise an error if no matching files are found
        if not filetype_files:
            raise FileNotFoundError(f"No files of type '{filetype}' were found at a depth of {max_depth} for the given filepath.")
        
        filetype_files.sort()

        # Return the list of matching filepaths
        return filetype_files



    ####################################################################################################
    ############################ Count filetypes in current directory ##################################
    ####################################################################################################



    def _count_files_in_dir(self, directory: str, filetype: str) -> int:
        """
        Count the number of files in a directory that have a specific file type.

        Parameters:
        - directory (str): The path to the directory to search in.
        - filetype (str): The file type (parameters, omega, nrg, etc.) to count.

        Returns:
        - int: The number of files in the specified directory that match the given file type.
        """
        # Check if the provided directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"The directory does not exist: {directory}")
        
        # Check if the provided path is a directory
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"The path given is not a directory: {directory}")

        filetype_count = 0
        
        # Iterate through the files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if the current item is a file and if its name starts with the specified file type
            if os.path.isfile(file_path) and filename.startswith(filetype):
                filetype_count += 1

        return filetype_count








# import concurrent.futures

# class GENEFileFinderParallel:
#     def __init__(self, input_filepath: str):
#         self.input_filepath = input_filepath

#     def find_files(self, filetype: str = 'parameters', max_depth: int = 3, smart_param_append: bool = True):
#         input_filepath_list = self.input_filepath if isinstance(self.input_filepath, list) else self.input_filepath.split(',')
#         filetype_files = []

#         # Define a nested function to process each path in parallel
#         def process_path(path):
#             result_files = []
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"The directory does not exist: {path}")
#             for root, dirs, files in os.walk(path):
#                 dirs[:] = [d for d in dirs if not any(d.startswith(prefix) for prefix in ['X_', 'in_par'])]
#                 current_depth = root[len(path):].count(os.sep)
#                 if current_depth <= max_depth:
#                     for file in files:
#                         if (self._count_files_in_dir(root, filetype) > 1) and smart_param_append:
#                             modifier = '_'
#                         else:
#                             modifier = ''
#                         if file.startswith(filetype + modifier):
#                             result_files.append(os.path.join(root, file))
#             return result_files

#         # Use ThreadPoolExecutor to parallelize directory traversal
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             results = executor.map(process_path, input_filepath_list)
        
#         for result in results:
#             filetype_files.extend(result)

#         if not filetype_files:
#             raise FileNotFoundError(f"No files of type '{filetype}' were found at a depth of {max_depth} for the given filepath.")
        
#         filetype_files.sort()
#         return filetype_files

#     def _count_files_in_dir(self, directory: str, filetype: str) -> int:
#         if not os.path.exists(directory):
#             raise FileNotFoundError(f"The directory does not exist: {directory}")
#         if not os.path.isdir(directory):
#             raise NotADirectoryError(f"The path given is not a directory: {directory}")
#         filetype_count = 0
#         for filename in os.listdir(directory):
#             if os.path.isfile(os.path.join(directory, filename)) and filename.startswith(filetype):
#                 filetype_count += 1
#         return filetype_count
