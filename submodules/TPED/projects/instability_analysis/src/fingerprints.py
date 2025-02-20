import os
import numpy as np
import pandas as pd

from TPED.projects.utils.read_write_geometry import read_geometry_local
from TPED.projects.utils.finite_differences import fd_d1_o4

from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GENEFileFinder as GFF
from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC
from TPED.projects.GENE_sim_reader.utils.fieldlib import fieldfile

from TPED.projects.GENE_sim_reader.src.GENE_nrg_data import GeneNrg as GN
from projects.GENE_sim_reader.archive.ARCHIVE_GENE_field_data import GeneField as GF
from TPED.projects.GENE_sim_reader.src.GENE_omega_data import GeneOmega as GO
from TPED.projects.GENE_sim_reader.src.GENE_parameters_data import GeneParameters as GP



input_nrg_quantities = ['Q_EM', 'Q_ES', 'Gamma_ES']
input_species = ['e', 'i']


def fingerprint_to_csv(fingerprint_list, output_filepath = None):
    """
    Convert a list of fingerprints to a CSV file.

    Args:
        fingerprint_list (list): A list of fingerprints.
        output_filepath (str, optional): The path to the output directory. If not provided, the CSV file will be saved in the current working directory.

    Returns:
        None

    This function takes a list of fingerprints and converts it into a pandas DataFrame. If the `output_filepath` is not provided, 
    the CSV file will be saved in the current working directory with the name 'fingerprint_data.csv'. 
    If the `output_filepath` is provided, the CSV file will be saved in the specified directory with the name 'fingerprint_data.csv'.

    Example:
        fingerprint_list = [{'diff': 1, 'phi_cont': 2, 'apar_cont': 3, 'diff/abs': 0.5}, {'diff': 2, 'phi_cont': 3, 'apar_cont': 4, 'diff/abs': 0.75}]
        fingerprint_to_csv(fingerprint_list, output_filepath='/path/to/output/directory')
        # The CSV file will be saved in '/path/to/output/directory/fingerprint_data.csv'
    """

    fingerprint_df = pd.DataFrame(fingerprint_list)
    if output_filepath is None:
        cwd = os.getcwd()
        save_filepath = os.path.join(cwd, 'fingerprint_data.csv')
    else:
        save_filepath = os.path.join(output_filepath, 'fingerprint_data.csv')
    fingerprint_df.to_csv(save_filepath)


def fingerprint_quantities(input_filepath):
    """
    Extracts fingerprint quantities from the given input file path and returns a list of dictionaries 
    containing various calculated quantities related to omega, nrg, and field values.
    :param input_filepath(str/list): The file path(s) to extract fingerprint quantities from.
    :return: A list of dictionaries, with each dictionary representing fingerprint quantities for a specific file.
    """
    
    fingerprint_list = []

    omega_paths = GFF(input_filepath).find_files('omega')
    for omega_path in omega_paths:
        omega_dict = GO(omega_path).omega_filepath_to_dict()

        if omega_dict:
            fingerprint_dict = {}

            param_path = GFC(omega_path).switch_suffix_file('parameters')
            param_dict = GP(param_path).parameters_filepath_to_dict()

            nrg_path = GFC(omega_path).switch_suffix_file('nrg')
            nrg_dict = GN(nrg_path).nrg_filepath_to_dict(input_nrg_species=input_species, 
                                                         input_nrg_quantities=input_nrg_quantities)

            directory = os.path.dirname(omega_path)
            suffix = GFC(omega_path).suffix_from_filepath()

            fingerprint_dict = calc_epar_quantities_temp(fingerprint_dict, param_dict, omega_dict, directory, suffix)

            # SIMULATION LOCATIONS
            fingerprint_dict['directory'] = directory
            fingerprint_dict['suffix'] = suffix

            # OMEGA VALUES
            fingerprint_dict['omega'] = omega_dict['omega']
            fingerprint_dict['gamma'] = omega_dict['gamma']

            # NRG VALUES
            fingerprint_dict['QEM/QES_e'] = nrg_dict['Q_EM_e'] / nrg_dict['Q_ES_e']
            fingerprint_dict['QEM/QES_i'] = nrg_dict['Q_EM_i'] / nrg_dict['Q_ES_i']
            fingerprint_dict['QESi/QESe'] = nrg_dict['Q_ES_i'] / nrg_dict['Q_ES_e']
            fingerprint_dict['QEMi/QEMe'] = nrg_dict['Q_EM_i'] / nrg_dict['Q_EM_e']
            fingerprint_dict['Gammae/Qe'] = nrg_dict['Gamma_ES_e'] / nrg_dict['Q_ES_e']

            for key in fingerprint_dict.keys():
                if isinstance(fingerprint_dict[key], np.ndarray):
                    fingerprint_dict[key] = float(fingerprint_dict[key])

            fingerprint_list.append(fingerprint_dict)

    return fingerprint_list











def calc_epar_quantities_temp(fingerprint_dict:dict, param_dict:dict, omega_dict:dict, directory:str, suffix:str):

    fingerprint_dict = fingerprint_dict.copy()

    mag_geom_path = os.path.join(directory, param_dict['magn_geometry'] + "_" +suffix)

    gpars,geometry = read_geometry_local(mag_geom_path)
    jacxB = geometry['gjacobian']*geometry['gBfield']
    
    omega = omega_dict['omega']
    gamma = omega_dict['gamma']
    if omega == 0:
        omega += 1e-10
    if gamma == 0:
        gamma += 1e-10

    omega_complex = (gamma + omega*(0.0+1.0J))
    omega_phase = np.log(omega_complex/np.abs(omega_complex))/(0.0+1.0J)
    # print("omega_complex",omega_complex)
    # print("omega_phase",omega_phase)

    field_filepath = GFC(omega_dict['filepath']).switch_suffix_file('field')

    field = fieldfile(field_filepath, param_dict)
    time = np.array(field.tfld)
    itime = -1
    
    # print("Looking at the mode structure at time:",time[itime])
    field.set_time(time[itime])

    ntot = field.nz*field.nx

    dz = float(2*field.nx)/ntot
    dz = float(2.0)/float(field.nz)
    zgrid = np.arange(ntot)/float(ntot-1)*(2*field.nx-dz)-field.nx
    #print 'zgrid',zgrid

    phi = np.zeros(ntot,dtype='complex128')
    apar = np.zeros(ntot,dtype='complex128')

    if 'n0_global' in param_dict.keys() and 'q0' in param_dict.keys():
        phase_fac = -np.e**(-2.0*np.pi*(0.0+1.0J)*param_dict['n0_global']*param_dict['q0'])
    else:
        phase_fac = -1.0

    if param_dict['shat'] < 0.0:
        for i in range(int(field.nx/2)+1):
            phi[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.phi()[:,0,-i]*phase_fac**i
            if i < int(field.nx/2):
                phi[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.phi()[:,0,i+1]*phase_fac**(-(i+1))
            if param_dict['n_fields']>1:
                apar[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.apar()[:,0,-i]*phase_fac**i
                if i < int(field.nx/2):
                    apar[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.apar()[:,0,i+1]*phase_fac**(-(i+1))
    else:
        for i in range(int(field.nx/2)):
            phi[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.phi()[:,0,i]*phase_fac**i
            if i < int(field.nx/2):
                phi[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.phi()[:,0,-1-i]*phase_fac**(-(i+1))
            if param_dict['n_fields']>1:
                apar[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.apar()[:,0,i]*phase_fac**i
                if i < int(field.nx/2):
                    apar[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.apar()[:,0,-1-i]*phase_fac**(-(i+1))
    
    # zavg=np.sum(np.abs(phi)*np.abs(zgrid))/np.sum(np.abs(phi))
    phi = phi/field.phi()[int(field.nz/2),0,0]
    apar = apar/field.phi()[int(field.nz/2),0,0]

    gradphi = fd_d1_o4(phi,zgrid)
    for i in range(param_dict['nx0']):
        gradphi[param_dict['nz0']*i:param_dict['nz0']*(i+1)] = gradphi[param_dict['nz0']*i:param_dict['nz0']*(i+1)]/jacxB[:]/np.pi
    
    diff = np.sum(np.abs(gradphi + omega_complex*apar))
    phi_cont = np.sum(np.abs(gradphi))
    apar_cont = np.sum(np.abs(omega_complex*apar))
    # print("diff",diff)
    # print("phi_cont",phi_cont)
    # print("apar_cont",apar_cont)
    # print("diff/abs",diff/(phi_cont+apar_cont))

    fingerprint_dict['diff'] = diff
    fingerprint_dict['phi_cont'] = phi_cont
    fingerprint_dict['apar_cont'] = apar_cont
    fingerprint_dict['diff/abs'] = diff/(phi_cont+apar_cont)

    return fingerprint_dict











def calc_epar_quantities(fingerprint_dict:dict, param_dict:dict, omega_dict:dict, directory:str, suffix:str):

    fingerprint_dict = fingerprint_dict.copy()

    mag_geom_path = os.path.join(directory, param_dict['magn_geometry'] + "_" +suffix)
    gpars,geometry = read_geometry_local(mag_geom_path)

    jacxB = geometry['gjacobian']*geometry['gBfield']

    omega = omega_dict['omega']
    gamma = omega_dict['gamma']
    if omega == 0:
        omega += 1e-10
    if gamma == 0:
        gamma += 1e-10
    
    omega_complex = (gamma + omega*(0.0+1.0J))
    omega_phase = np.log(omega_complex/np.abs(omega_complex))/(0.0+1.0J)

    
    field_path = GFC(omega_dict['filepath']).switch_suffix_file('field')
    field_dict = GF(field_path).field_filepath_to_dict()

    print(field_dict.keys())
    print(field_dict['time'])

    zgrid = np.array(field_dict['zgrid'])
    phi = np.array(field_dict['field_phi'][-1]) # access last time slice of phi
    
    phase_array = np.empty(len(zgrid))
    phase_array[:] = np.real(omega_phase)

    print('phi',phi.shape, phi)
    print('zgrid',zgrid.shape, zgrid)
    gradphi = fd_d1_o4(phi,zgrid)

    print("gradphi",gradphi.shape, gradphi)
    print('unique gradphi',np.unique(gradphi))
    for i in range(param_dict['nx0']):
        upper_bound = param_dict['nz0']*i
        lower_bound = param_dict['nz0']*(i+1)
        gradphi[upper_bound:lower_bound] = gradphi[upper_bound:lower_bound]/jacxB[:]/np.pi

    diff = np.sum(np.abs(gradphi + omega_complex*apar))
    phi_cont = np.sum(np.abs(gradphi))
    apar_cont = np.sum(np.abs(omega_complex*apar))

    fingerprint_dict['diff'] = diff
    fingerprint_dict['phi_cont'] = phi_cont
    fingerprint_dict['apar_cont'] = apar_cont
    fingerprint_dict['diff/abs'] = diff/(phi_cont+apar_cont)

    return fingerprint_dict




import time
import concurrent.futures

def process_file_parallel(filepaths):
    def process_file(gene_filepath):
        # print(f"Processing {gene_filepath} in parallel...")
        result = field_resolution_check(gene_filepath, plot_check=False, save_data=True, overwrite=True)
        # print("Done with:", gene_filepath)
        return result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, filepaths))
    return results
