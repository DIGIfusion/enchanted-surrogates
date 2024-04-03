from .base import Parser
import subprocess
import os
import numpy as np
from typing import List
from copy import deepcopy

class TGLFparser(Parser):
    """ An I/O parser for TGLF """
    def __init__(self):
        self.ky_spectrum_file = 'out.tglf.ky_spectrum'
        self.growth_rate_freq_file = 'out.tglf.eigenvalue_spectrum'
        self.flux_spectrum_file = 'out.tglf.sum_flux_spectrum'
        self.default_parameters = self.input_dict()

    def write_input_file(self, params: dict, run_dir: str):
        # give some parameters write to a new input file!
        # TODO: write a standard input file based on somthing?
        print('Writing to', run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'input.tglf')
            subprocess.run(['touch', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')

        default_params: dict = deepcopy(self.default_parameters)
        
        with open(input_fpath, 'w') as file:
            for param_name, val in params.items():
                default_params.pop(param_name)
                file.write(f'{param_name}={val}\n')
            for default_param_name, default_param_dict in default_params.items():
                default_val = default_param_dict['default']
                file.write(f'{default_param_name}={default_val}\n')
        # TODO: check for input comparisons with available inputs for code?

    def read_output_file(self, run_dir: str):
        ky_spectrum_file_path = os.path.join(run_dir, self.ky_spectrum_file)
        growth_rate_freq_file_path = os.path.join(
            run_dir, self.growth_rate_freq_file)
        flux_spectrum_file_path = os.path.join(
            run_dir, self.flux_spectrum_file)

        self.ky_spectrum = np.genfromtxt(
            ky_spectrum_file_path, dtype=None, skip_header=2)
        self.eigenvalue_spectrum = np.genfromtxt(
            growth_rate_freq_file_path, dtype=None, skip_header=2)
        self.flux_spectrums = self.parse_flux_spectrum(flux_spectrum_file_path)
        self.fluxes = []
        for species in self.flux_spectrums:
            energy_flux = species[:, 1].sum()
            particle_flux = species[:, 0].sum()
            self.fluxes.extend([energy_flux, particle_flux])
        # self.fluxes = [flux_spec.sum() for flux_spec in self.flux_spectrums]

    def parse_flux_spectrum(self, file_path) -> List[np.ndarray]:
        data_sets = []
        current_data_set = []
        with open(file_path, 'r') as file:
            for line in file:
                # Check if the line is a species marker indicating
                # a new data set
                if line.startswith(' species ='):
                    # If we already have data collected, convert it to a NumPy
                    # array and reset for the next set
                    if current_data_set:
                        data_sets.append(
                            np.array(current_data_set, dtype=float))
                        current_data_set = []
                    # Skip the next line which contains column names
                    next(file)
                else:
                    # Collect data lines into the current set
                    data_values = line.split()
                    if data_values:  # Ensure it's not an empty line
                        current_data_set.append(
                            [float(value) for value in data_values])
            # Don't forget to add the last set if the file ends without
            # a new marker
            if current_data_set:
                data_sets.append(np.array(current_data_set, dtype=float))
        return data_sets

    def input_dict(self, ) -> dict: 
        parameters_dict = self.default_ga_input()

        # BASED ON ASTRA INTERFACE FILE, ALL SET BEFORE THE RADIAL LOOP 
        parameters_dict["KYGRID_MODEL"]['default'] = "4"
        parameters_dict["SAT_RULE"]['default'] = "2"
        parameters_dict["XNU_MODEL"]['default'] = "3" # SAT_RULE 2
        parameters_dict["ALPHA_ZF"] = {'interface_parameter': 'tglf_alpha_zf_in', 'default': "1"} # SAT_RULE 2
        
        parameters_dict['USE_BPER']['default'] = ".True."
        parameters_dict['USE_MHD_RULE']['default'] = ".Frue."

        parameters_dict['NMODES']['default'] = str(int(parameters_dict['NS']['default']) + 2)
        parameters_dict['NBASIS_MAX']['default'] = 6
        parameters_dict['NKY']['default'] = 19
        parameters_dict['UNITS']['default'] = 'CGYRO'

        parameters_dict['VPAR_SHEAR_MODEL']['default'] = 1
        

        # ======================= BEGIN UNSURE =======================
        # parameters_dict["WDIA_TRAP"] = {'interface_parameter': 'tglf_wdia_trapped_in', 'default': "1"} # SAT_RULE 2
        # tglf_use_ave_ion_grid_in = '.true.'
        # tglf_dump_flag_in     = .False.   ! Dumps input file
        # tglf_test_flag_in     = 0
        # tglf_nn_max_error_in  = 0
        # tglf_damp_psi_in       = 0.
        # tglf_damp_sig_in       = 0.
        # ======================= END UNSURE =======================

        return parameters_dict
    def default_ga_input(self,) -> dict: 
        parameters_dict = {
            "USE_TRANSPORT_MODEL": {"interface_parameter": "tglf_use_transport_model_in", "default": ".true."},
            "USE_BPER": {"interface_parameter": "tglf_use_bper_in", "default": ".false."},
            "USE_BPAR": {"interface_parameter": "tglf_use_bpar_in", "default": ".false."},
            "USE_BISECTION": {"interface_parameter": "tglf_use_bisection_in", "default": ".true."},
            "USE_MHD_RULE": {"interface_parameter": "tglf_use_mhd_rule_in", "default": ".true."},
            "USE_INBOARD_DETRAPPED": {"interface_parameter": "tglf_use_inboard_detrapped_in", "default": ".false."},
            "SAT_RULE": {"interface_parameter": "tglf_sat_rule_in", "default": "0"},
            "KYGRID_MODEL": {"interface_parameter": "tglf_kygrid_model_in", "default": "1"},
            "XNU_MODEL": {"interface_parameter": "tglf_xnu_model_in", "default": "2"},
            "VPAR_MODEL": {"interface_parameter": "tglf_vpar_model_in", "default": "0"},
            "VPAR_SHEAR_MODEL": {"interface_parameter": "tglf_vpar_shear_model_in", "default": "0"},
            "SIGN_BT": {"interface_parameter": "tglf_sign_bt_in", "default": "1.0"},
            "SIGN_IT": {"interface_parameter": "tglf_sign_it_in", "default": "1.0"},
            "KY": {"interface_parameter": "tglf_ky_in", "default": "0.3"},
            "NEW_EIKONAL": {"interface_parameter": "tglf_new_eikonal_in", "default": ".true."},
            "VEXB": {"interface_parameter": "tglf_vexb_in", "default": "0.0"},
            "VEXB_SHEAR": {"interface_parameter": "tglf_vexb_shear_in", "default": "0.0"},
            "BETAE": {"interface_parameter": "tglf_betae_in", "default": "0.0"},
            "XNUE": {"interface_parameter": "tglf_xnue_in", "default": "0.0"},
            "ZEFF": {"interface_parameter": "tglf_zeff_in", "default": "1.0"},
            "DEBYE": {"interface_parameter": "tglf_debye_in", "default": "0.0"},
            "IFLUX": {"interface_parameter": "tglf_iflux_in", "default": ".true."},
            "IBRANCH": {"interface_parameter": "tglf_ibranch_in", "default": "-1"},
            "NMODES": {"interface_parameter": "tglf_nmodes_in", "default": "2"},
            "NBASIS_MAX": {"interface_parameter": "tglf_nbasis_max_in", "default": "4"},
            "NBASIS_MIN": {"interface_parameter": "tglf_nbasis_min_in", "default": "2"},
            "NXGRID": {"interface_parameter": "tglf_nxgrid_in", "default": "16"},
            "NKY": {"interface_parameter": "tglf_nky_in", "default": "12"},
            "ADIABATIC_ELEC": {"interface_parameter": "tglf_adiabatic_elec_in", "default": ".false."},
            "ALPHA_P": {"interface_parameter": "tglf_alpha_p_in", "default": "1.0"},
            "ALPHA_MACH": {"interface_parameter": "tglf_alpha_mach_in", "default": "0.0"},
            "ALPHA_E": {"interface_parameter": "tglf_alpha_e_in", "default": "1.0"},
            "ALPHA_QUENCH": {"interface_parameter": "tglf_alpha_quench_in", "default": "0.0"},
            "XNU_FACTOR": {"interface_parameter": "tglf_xnu_factor_in", "default": "1.0"},
            "DEBYE_FACTOR": {"interface_parameter": "tglf_debye_factor_in", "default": "1.0"},
            "ETG_FACTOR": {"interface_parameter": "tglf_etg_factor_in", "default": "1.25"},
            "WRITE_WAVEFUNCTION_FLAG": {"interface_parameter": "tglf_write_wavefunction_flag_in", "default": "0"},
            "UNITS": {"interface_parameter": "units_in", "default": "GYRO"},
        }

        species_parameters = {
            "NS": {"interface_parameter": "tglf_ns_in", "default": "2"},
            "ZS_1": {"interface_parameter": "tglf_zs_in(1)", "default": "-1.0"},
            "MASS_1": {"interface_parameter": "tglf_mass_in(1)", "default": "2.723e-4"},
            "RLNS_1": {"interface_parameter": "tglf_rlns_in(1)", "default": "1.0"},
            "RLTS_1": {"interface_parameter": "tglf_rlts_in(1)", "default": "3.0"},
            "TAUS_1": {"interface_parameter": "tglf_taus_in(1)", "default": "1.0"},
            "AS_1": {"interface_parameter": "tglf_as_in(1)", "default": "1.0"},
            "VPAR_1": {"interface_parameter": "tglf_vpar_in(1)", "default": "0.0"},
            "VPAR_SHEAR_1": {"interface_parameter": "tglf_vpar_shear_in(1)", "default": "0.0"},
            "ZS_2": {"interface_parameter": "tglf_zs_in(2)", "default": "1.0"},
            "MASS_2": {"interface_parameter": "tglf_mass_in(2)", "default": "1.0"},
            "RLNS_2": {"interface_parameter": "tglf_rlns_in(2)", "default": "1.0"},
            "RLTS_2": {"interface_parameter": "tglf_rlts_in(2)", "default": "3.0"},
            "TAUS_2": {"interface_parameter": "tglf_taus_in(2)", "default": "1.0"},
            "AS_2": {"interface_parameter": "tglf_as_in(2)", "default": "1.0"},
            "VPAR_2": {"interface_parameter": "tglf_vpar_in(2)", "default": "0.0"},
            "VPAR_SHEAR_2": {"interface_parameter": "tglf_vpar_shear_in(2)", "default": "0.0"},
        }

        guassian_width_parameters = {
        'WIDTH': {'interface_parameter': 'tglf_width_in', 'default': '1.65'},
        'WIDTH_MIN': {'interface_parameter': 'tglf_width_min_in', 'default': '0.3'},
        'NWIDTH': {'interface_parameter': 'tglf_nwidth_in', 'default': '21'},
        'FIND_WIDTH': {'interface_parameter': 'tglf_find_width_in', 'default': '.true.'}
        }

        miller_geometry_parameters = {
            "GEOMETRY_FLAG": {"interface_parameter": "tglf_geometry_flag_in", "default": "1"},
            'RMIN_LOC': {'interface_parameter': 'tglf_rmin_loc_in', 'default': '0.5'},
            'RMAJ_LOC': {'interface_parameter': 'tglf_rmaj_loc_in', 'default': '3.0'},
            'ZMAJ_LOC': {'interface_parameter': 'tglf_zmaj_loc_in', 'default': '0.0'},
            'Q_LOC': {'interface_parameter': 'tglf_q_loc_in', 'default': '2.0'},
            'Q_PRIME_LOC': {'interface_parameter': 'tglf_q_prime_loc_in', 'default': '16.0'},
            'P_PRIME_LOC': {'interface_parameter': 'tglf_p_prime_loc_in', 'default': '0.0'},
            'DRMINDX_LOC': {'interface_parameter': 'tglf_drmindx_loc_in', 'default': '1.0'},
            'DRMAJDX_LOC': {'interface_parameter': 'tglf_drmajdx_loc_in', 'default': '0.0'},
            'DZMAJDX_LOC': {'interface_parameter': 'tglf_dzmajdx_loc_in', 'default': '0.0'},
            'KAPPA_LOC': {'interface_parameter': 'tglf_kappa_loc_in', 'default': '1.0'},
            'S_KAPPA_LOC': {'interface_parameter': 'tglf_s_kappa_loc_in', 'default': '0.0'},
            'DELTA_LOC': {'interface_parameter': 'tglf_delta_loc_in', 'default': '0.0'},
            'S_DELTA_LOC': {'interface_parameter': 'tglf_s_delta_loc_in', 'default': '0.0'},
            'ZETA_LOC': {'interface_parameter': 'tglf_zeta_loc_in', 'default': '0.0'},
            'S_ZETA_LOC': {'interface_parameter': 'tglf_s_zeta_loc_in', 'default': '0.0'},
            'KX0_LOC': {'interface_parameter': 'tglf_kx0_in', 'default': '0.0'}
            }

        s_alpha_parameters = {
            "RMIN_SA": {"interface_parameter": "tglf_rmin_sa_in", "default": "0.5"},
            "RMAJ_SA": {"interface_parameter": "tglf_rmaj_sa_in", "default": "3.0"},
            "Q_SA": {"interface_parameter": "tglf_q_sa_in", "default": "2.0"},
            "SHAT_SA": {"interface_parameter": "tglf_shat_sa_in", "default": "1.0"},
            "ALPHA_SA": {"interface_parameter": "tglf_alpha_sa_in", "default": "0.0"},
            "XWELL_SA": {"interface_parameter": "tglf_xwell_sa_in", "default": "0.0"},
            "THETA0_SA": {"interface_parameter": "tglf_theta0_sa_in", "default": "0.0"},
            "B_MODEL_SA": {"interface_parameter": "tglf_b_model_sa_in", "default": "1"},
            "FT_MODEL_SA": {"interface_parameter": "tglf_ft_model_sa_in", "default": "1"}
        }

        change_at_own_risk_parameters = {
            "THETA_TRAPPED": {"interface_parameter": "tglf_theta_trapped_in", "default": "0.7"},
            "PARK": {"interface_parameter": "tglf_park_in", "default": "1.0"},
            "GHAT": {"interface_parameter": "tglf_ghat_in", "default": "1.0"},
            "GCHAT": {"interface_parameter": "tglf_gchat_in", "default": "1.0"},
            "WD_ZERO": {"interface_parameter": "tglf_wd_zero_in", "default": "0.1"},
            "LINSKER_FACTOR": {"interface_parameter": "tglf_linsker_factor_in", "default": "0.0"},
            "GRADB_FACTOR": {"interface_parameter": "tglf_gradB_factor_in", "default": "0.0"},
            "FILTER": {"interface_parameter": "tglf_filter_in", "default": "2.0"}
        }

        parameters_dict = {**parameters_dict, **change_at_own_risk_parameters, **s_alpha_parameters, **miller_geometry_parameters, **guassian_width_parameters, **species_parameters}
        return parameters_dict