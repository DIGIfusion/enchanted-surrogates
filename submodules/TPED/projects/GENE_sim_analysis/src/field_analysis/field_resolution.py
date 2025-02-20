import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle

from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


from TPED.projects.GENE_sim_reader.archive.ARCHIVE_GENE_field_data import GeneField as GF
from TPED.projects.GENE_sim_reader.utils.GENE_filepath_converter import GeneFilepathConverter as GFC


from TPED.projects.GENE_sim_reader.src.GENE_field_data import GeneField as GF
from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder as GFF

import os


def plot_para_mag_potential(filepath_list:str|list):    
    if isinstance(filepath_list, str):
        if os.path.basename(filepath_list).startswith('field'):
            filepath_list = [filepath_list]
        elif os.path.isdir(filepath_list):
            filepath_list = GFF(filepath_list).find_files('field')
    elif isinstance(filepath_list, list):
        if all([os.path.basename(filepath).startswith('field') for filepath in filepath_list]):
            pass
        else:
            raise ValueError('Invalid input. Please provide a list of field filepaths or a directory path.')   
    else:
        raise ValueError('Invalid input. Please provide a list of field filepaths or a directory path.')



    for field_filepath in filepath_list:
        field = GF(field_filepath)
        field_dict = field.field_filepath_to_dict(time_criteria='last')

        zgrid = field_dict['zgrid']
        apar = field_dict['field_apar'][-1]

        plt.title(f'$A_{{\\parallel}}$')
        plt.plot(zgrid,np.real(apar),color='red',label=f'$Re[A_{{\\parallel}}]$')
        plt.plot(zgrid,np.imag(apar),color='blue',label=f'$Im[A_{{\\parallel}}]$')
        plt.plot(zgrid,np.abs(apar),color='black',label=f'$|A_{{\\parallel}}|$')
        plt.xlabel(r'$z/\pi$',size=18)
        plt.legend()
        plt.show()


def plot_mode_structure(filepath_list:str|list):
    
    
    if isinstance(filepath_list, str):
        if os.path.basename(filepath_list).startswith('field'):
            filepath_list = [filepath_list]
        elif os.path.isdir(filepath_list):
            filepath_list = GFF(filepath_list).find_files('field')
    elif isinstance(filepath_list, list):
        if all([os.path.basename(filepath).startswith('field') for filepath in filepath_list]):
            pass
        else:
            raise ValueError('Invalid input. Please provide a list of field filepaths or a directory path.')   
    else:
        raise ValueError('Invalid input. Please provide a list of field filepaths or a directory path.')



    for field_filepath in filepath_list:
        field = GF(field_filepath)
        field_dict = field.field_filepath_to_dict(time_criteria='last')

        zgrid = field_dict['zgrid']
        phi = field_dict['field_phi'][-1]

        plt.title(r'$\phi$')
        plt.plot(zgrid,np.real(phi),color='red',label=r'$Re[\phi]$')
        plt.plot(zgrid,np.imag(phi),color='blue',label=r'$Im[\phi]$')
        plt.plot(zgrid,np.abs(phi),color='black',label=r'$|\phi|$')
        plt.xlabel(r'$z/\pi$',size=18)
        plt.legend()
        plt.show()








GENE_SIM_FOLDER = 'X_GENE_sim_extra_data'


def field_resolution_check(field_filepath, plot_check=False, save_data=False, overwrite=False, visual_check=False):

    dir_path = os.path.dirname(field_filepath)
    save_path = os.path.join(dir_path, GENE_SIM_FOLDER)
    suffix = GFC(field_filepath).suffix_from_filepath()
    pkl_filename = f'field_resolution_data_{suffix}.pkl'




    



    if os.path.exists(os.path.join(save_path, pkl_filename)) and not overwrite:
        # print('Loading data from pickle file')
        with open(os.path.join(save_path, pkl_filename), 'rb') as pickle_file:
            fft_data_dict = pickle.load(pickle_file)

    else:
        # print('Generating field data and FFT')
        field = GF(field_filepath)
        field_dict = field.field_filepath_to_dict(time_criteria='last', input_fields = ['field_phi',  'field_apar'])
        field_xarray = field.field_dict_to_xarray(field_dict)

        # Selecting data at the last time index
        last_time_field_data = field_xarray.field_phi.isel(time=-1)

        fft_data_dict = fft_calculations(last_time_field_data)
        fft_data_dict['last_field_data'] = np.array(last_time_field_data)


    if plot_check:
        plot_field_resolution(fft_data_dict['last_field_data'], fft_data_dict['freq'], fft_data_dict['norm_fft_magnitude'])



    vis_res = fft_data_dict.get('visual_resolution', None)
    if overwrite:
        vis_res = None

    if visual_check and (vis_res is None):

        clear_output(wait=True)  # Clear the output before plotting

        plt.figure()  # Start with a new figure to clear previous plots
        plot_field_resolution_SIMPLE(fft_data_dict['last_field_data'])

        category = ''
        categories = ['1', '2', '3']
        category_names = {'1': 'Resolved', '2': 'Partially Resolved', '3': 'Unresolved'}

        while category not in categories:
            category = input(f"Select a category by Pressing 1 ({category_names['1']}), 2 ({category_names['2']}), or 3 ({category_names['3']}):")

        fft_data_dict['visual_resolution'] = category_names[category]
        print(f"Selected category: {fft_data_dict['visual_resolution']}")

        time.sleep(0.3)
        plt.close()  # Clear the plot to prevent overlapping




    if save_data:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_file = os.path.join(save_path, pkl_filename)

        # print('Saving data to pickle file')
        print(f'Saving file to: {save_file}')
        
        time.sleep(0.5)

        with open(save_file, 'wb') as pickle_file:
            pickle.dump(fft_data_dict, pickle_file)


    return fft_data_dict







def fft_calculations(field_data):

    fft_data_dict = {}

    # Perform FFT on the complex data
    fft_result = np.fft.fft(field_data)
    freq = np.fft.fftfreq(len(field_data))
    fft_magnitude = np.abs(fft_result)
    norm_fft_magnitude = fft_magnitude / np.sum(fft_magnitude)

    fft_data_dict['freq'] = freq
    fft_data_dict['norm_fft_magnitude'] = norm_fft_magnitude

    half_freq = max(freq) / 2

    # Indices for low frequency band and high frequency band
    low_freq_indices = np.logical_and(freq > -half_freq, freq < half_freq)
    high_freq_indices = np.logical_not(low_freq_indices)

    # Energy calculation for low and high frequency bands
    low_freq_energy = np.sum(norm_fft_magnitude[low_freq_indices]**2)
    high_freq_energy = np.sum(norm_fft_magnitude[high_freq_indices]**2)

    fft_data_dict['norm_energy_ratio'] = low_freq_energy / (low_freq_energy + high_freq_energy)
    
    energy_ratio = high_freq_energy / low_freq_energy
    fft_data_dict['energy_ratio'] = energy_ratio
    fft_data_dict['log_energy_ratio'] = np.log(energy_ratio + 1)

    # Calculate upper 95% confidence interval for positive and negative frequencies
    pos_freq_indices = freq > 0
    neg_freq_indices = freq < 0
    pos_upper_95_freq = calc_95_confidence_interval(freq[pos_freq_indices], fft_magnitude[pos_freq_indices])
    neg_upper_95_freq = calc_95_confidence_interval(freq[neg_freq_indices], fft_magnitude[neg_freq_indices])
    fft_data_dict['pos_upper_95_freq'] = pos_upper_95_freq
    fft_data_dict['neg_upper_95_freq'] = neg_upper_95_freq
    fft_data_dict['upper_95_dist'] = pos_upper_95_freq - neg_upper_95_freq

    # Calculate Sobolev norm
    sobolev_norm_k2_p1 = compute_sobolev_norm(fft_result, freq, k=2, p=1)
    fft_data_dict['sobolev_norm_k2_p1'] = sobolev_norm_k2_p1

    return fft_data_dict





import numpy as np

def compute_sobolev_norm(fft_result, freq, k, p):
    """
    Compute the Sobolev norm of order k and integrability p for a given complex array.
    
    Parameters:
    data (np.ndarray): Input array of complex points.
    k (int): Order of the Sobolev space.
    p (int): Integrability parameter.
    
    Returns:
    float: Sobolev norm of the input data.
    """
    
    # Compute the Sobolev norm
    sobolev_norm = np.sum((1 + np.abs(freq)**2)**(k / 2) * np.abs(fft_result)**p) ** (1 / p)
    
    return sobolev_norm








def calc_95_confidence_interval(freq_slice, fft_magnitude_slice, n_samples=1000):

    norm_fft_magnitude = fft_magnitude_slice / np.sum(fft_magnitude_slice)

    # Bootstrapping to estimate 95% confidence interval
    bootstrap_means = []
    for _ in range(n_samples):  # number of bootstrap samples
        sample_indices = np.random.choice(np.arange(len(freq_slice)), size=len(freq_slice), replace=True, p=norm_fft_magnitude)
        sample = freq_slice[sample_indices]
        bootstrap_means.append(np.mean(sample))

    x_upper_95 = np.percentile(bootstrap_means, 95)

    return x_upper_95






def plot_field_resolution(last_time_field_data, freq, norm_fft_magnitude=None):

    # Extract the real, imaginary parts, and magnitude
    real_part = last_time_field_data.real
    imag_part = last_time_field_data.imag
    magnitude = np.abs(last_time_field_data)

    # Setting up the seaborn style
    sns.set(style="whitegrid")

    # Create a figure with subplots in a 2x2 configuration
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plotting real part
    sns.lineplot(x=np.arange(len(real_part)), y=real_part, ax=axes[0, 0])
    axes[0, 0].set_title('Real Part')

    # Plotting imaginary part
    sns.lineplot(x=np.arange(len(imag_part)), y=imag_part, ax=axes[0, 1])
    axes[0, 1].set_title('Imaginary Part')

    # Plotting magnitude of the data
    sns.lineplot(x=np.arange(len(magnitude)), y=magnitude, ax=axes[1, 0])
    axes[1, 0].set_title('Magnitude')

    # Plotting FFT results
    sns.lineplot(x=freq, y=norm_fft_magnitude, ax=axes[1, 1])
    axes[1, 1].set_title('FFT Magnitude')

    # Improve layout and display the plots
    plt.tight_layout()
    plt.show()




def plot_field_resolution_SIMPLE(last_time_field_data):

    # Extract the real, imaginary parts, and magnitude
    real_part = last_time_field_data.real
    imag_part = last_time_field_data.imag

    # Setting up the seaborn style
    sns.set(style="whitegrid")

    # Create a figure with subplots in a 1x2 configuration
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plotting real part
    sns.lineplot(x=np.arange(len(real_part)), y=real_part, ax=axes[0])
    axes[0].set_title('Real Part')

    # Plotting imaginary part
    sns.lineplot(x=np.arange(len(imag_part)), y=imag_part, ax=axes[1])
    axes[1].set_title('Imaginary Part')

    # Improve layout and display the plots
    plt.tight_layout()
    plt.show()





