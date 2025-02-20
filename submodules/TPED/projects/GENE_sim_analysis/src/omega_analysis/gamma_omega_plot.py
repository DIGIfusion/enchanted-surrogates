
import matplotlib.plt as plt
import xarray as xr
import pandas as pd

from TPED.projects.GENE_sim_reader.src.GENE_sim_data import GeneSimulation as GS




# plotting_dict = {
#     'nz0': {'color': 'blue', 'plot_type': 'bar'},
#     'hyp_z': {'plot_type': 'line'}
# }


def plot_gamma_omega(simulation_obj_list, plotting_dict:dict, simulation_labels:list = None):

    if not isinstance(simulation_obj_list, list):
        simulation_obj_list = [simulation_obj_list]

    # Plotting
    plt.figure(figsize=(14, 8))

    clean_sim_obj_list = []
    for simulation_obj in simulation_obj_list:
        if isinstance(simulation_obj, dict):
            simulation_obj = GS(simulation_obj)
        if not isinstance(simulation_obj, xr.Dataset):
            raise ValueError('simulation_obj must be an xarray Dataset object')


        
        # Plot for gamma
        plt.subplot(2, 3, 1)
        plt.scatter(simulation_obj['kymin'], simulation_obj['gamma'])
        plt.xlabel('kymin')
        plt.ylabel('Gamma')
        plt.title('Gamma vs kymin')
        plt.xscale('log')  # Setting x-axis to log scale
        plt.legend()

        # Plot for omega
        plt.subplot(2, 3, 2)
        plt.scatter(simulation_obj['kymin'], simulation_obj['omega'])
        plt.xlabel('kymin')
        plt.ylabel('Omega')
        plt.title('Omega vs kymin')
        plt.xscale('log')  # Setting x-axis to log scale
        plt.legend()

