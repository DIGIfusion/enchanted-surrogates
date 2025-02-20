#!/usr/bin/env python

import os
import optparse as op
import numpy as np
import matplotlib.pyplot as plt

# from TPED.projects.GENE_sim_reader.utils.X_file_functions import switch_suffix_file
from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC
from TPED.projects.GENE_sim_reader.utils.find_GENE_files import find_files
from TPED.projects.GENE_sim_reader.utils.fieldlib import fieldfile
from TPED.projects.GENE_sim_reader.src.GENE_parameters_data import parameters_filepath_to_dict
from TPED.projects.GENE_sim_reader.src.GENE_omega_data import omega_filepath_to_dict



# TODO: Rewrite to use field xarray functionality
# TODO: Compare field growth rate with omega_ growth rates (useful for training data for ML model for predicting omega from field data)


# def scanfile_info(scanfile_path:str):
#     param_filepaths = find_files(scanfile_path, 'parameters')

#     for param_path in param_filepaths:
        
#         omega_path = switch_suffix_file(param_path, 'omega')
#         omega_dict = omega_filepath_to_dict(omega_path)
#         gamma = omega_dict.get('gamma', None)

#         if gamma is None:
#             print('No gamma found in omega file. Calculating omega from field...')



####################################################################################################
########################## Calculate gamma and omega from field files ##############################
####################################################################################################



def calc_omega_from_field(omega_path:str, calc_phi:bool=True, calc_apar:bool=True):
        
    field_path = GFC.switch_suffix_file(omega_path, 'field')
    time, phi, apar = get_field_info_last_percent_time(field_path)

    omega_gr_dict, gamma_gr_dict = calc_plot_growth_rates(time, phi, apar, calc_phi=calc_phi, calc_apar=calc_apar)






####################################################################################################
########################## Collect relevant field info #############################################
####################################################################################################

    
def get_field_info_last_percent_time(field_path:str, end_time_percent:float = 0.1):
    """
    Extracts time slices and field data for the specified last percentage of simulation time.

    Args:
    field_path (str): Path to the field data file.
    end_time_percent (float, optional): Fraction of the end portion of the time data to analyze. Defaults to 0.1.

    Returns:
    tuple: A tuple containing the time slice array, phi field array, and apar field array (if applicable).
    """
    # Load parameter dictionary from a file with modified suffix
    param_path = GFC.switch_suffix_file(field_path, 'parameters')    
    param_dict = parameters_filepath_to_dict(param_path)
    
    # Load the field data using the parameter dictionary
    field_file = fieldfile(field_path, param_dict)
    time_array = np.array(field_file.tfld)

    # Determine the indices for the last end_time_percent of the data
    ind_start = int(time_array.shape[0] * (1 - end_time_percent))
    ind_end = len(time_array) - 1
    time_slice = time_array[ind_start:ind_end]

    # Initialize empty arrays for phi and potentially apar
    phi = np.empty(0, dtype='complex128')
    phi_ind_tuple = np.unravel_index(np.argmax(abs(field_file.phi()[:, 0, :])), (field_file.nz, field_file.nx))

    # Handle the apar field if more than one field is present
    apar = np.empty(0, dtype='complex128') if param_dict['n_fields'] > 1 else None
    if apar is not None:
        apar_ind_tuple = np.unravel_index(np.argmax(abs(field_file.apar()[:, 0, :])), (field_file.nz, field_file.nx))

    # Loop through the time indices and append field values
    for ind in range(ind_start, ind_end):
        field_file.set_time(field_file.tfld[ind])
        phi = np.append(phi, field_file.phi()[phi_ind_tuple[0], 0, phi_ind_tuple[1]])

        if apar is not None:
            apar = np.append(apar, field_file.apar()[apar_ind_tuple[0], 0, apar_ind_tuple[1]])

    return time_slice, phi, apar




####################################################################################################
########################## Calculate growth rates and frequencies ##################################
####################################################################################################


def calc_plot_growth_rates(time, phi, apar=None, plot_growth_rates:bool=True, calc_phi:bool=True, calc_apar:bool=True):
    """
    Calculate and optionally plot the growth rates derived from phi and apar fields over time.

    Args:
    time (np.array): Array of time points corresponding to the field data.
    phi (np.array): Array of phi field values.
    apar (np.array, optional): Array of apar field values. Defaults to None.
    plot_growth_rates (bool, optional): Flag to determine if the growth rates should be plotted. Defaults to True.

    Returns:
    dict: A dictionary containing growth rate data ('omega' and 'gamma') for phi and possibly apar.
    """
    omega_gr_dict = {}
    gamma_gr_dict = {}
    time_adj = np.delete(time, 0)  # Adjust time array by removing the first element

    def compute_omega(field_data):
        "Compute the growth rate omega for a given field data array."
        if len(field_data) < 2:
            return np.array([0.0 + 0.0j])
        omega = np.log(field_data / np.roll(field_data, 1)) / (time - np.roll(time, 1))
        return np.delete(omega, 0)  # delete the first element which is invalid


    comp_omega_phi = compute_omega(phi)
    gamma_phi = np.average(np.real(comp_omega_phi))
    omega_phi = np.average(np.imag(comp_omega_phi))
    omega_gr_dict['phi'] = omega_phi
    gamma_gr_dict['phi'] = gamma_phi

    if apar is not None:
        comp_omega_apar = compute_omega(apar)
        gamma_apar = np.average(np.real(comp_omega_apar))
        omega_apar = np.average(np.imag(comp_omega_apar))
        omega_gr_dict['apar'] = omega_apar
        gamma_gr_dict['apar'] = gamma_apar
        
        if plot_growth_rates and calc_apar:
            print("APAR - Gamma:", gamma_apar)
            print("APAR - Omega:", omega_apar)
            plt.plot(time_adj, np.real(comp_omega_apar), 'r--', label='gamma_apar')
            plt.plot(time_adj, np.imag(comp_omega_apar), 'b--', label='omega_apar')

    if plot_growth_rates:

        if calc_phi:
            print("PHI - Gamma:", gamma_phi)
            print("PHI - Omega:", omega_phi)
            plt.plot(time_adj, np.real(comp_omega_phi), 'orange', label='gamma_phi')
            plt.plot(time_adj, np.imag(comp_omega_phi), 'c', label='omega_phi')
    
        plt.xlabel('t(a/cs)')
        plt.ylabel('omega(cs/a)')
        plt.legend(loc='upper left')
        plt.title('Growth Rates')
        plt.show()
    
    return omega_gr_dict, gamma_gr_dict















if __name__ == "__main__":

    cwd = os.getcwd()

    parser=op.OptionParser(description='Calculates mode information and synthesizes scan info.')
    options,args=parser.parse_args()
    if len(args)!=1:
        exit("""
    Please include scan number as argument (e.g., 0001)."
        \n""")
    suffix = args[0]

    omega_path = os.path.join(cwd, 'omega_'+suffix)
    calc_omega_from_field(omega_path)

    # scanfile_path = os.path.join(cwd, 'scanfiles'+suffix)

    # scanfile_info(scanfile_path)
    