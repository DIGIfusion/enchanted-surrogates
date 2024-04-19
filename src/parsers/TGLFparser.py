from .base import Parser
import subprocess
import os
import numpy as np
from typing import List
from copy import deepcopy
import numpy as np 

class TGLFparser(Parser):
    """ An I/O parser for TGLF """
    def __init__(self):
        self.ky_spectrum_file = 'out.tglf.ky_spectrum'
        self.growth_rate_freq_file = 'out.tglf.eigenvalue_spectrum'
        self.flux_spectrum_file = 'out.tglf.sum_flux_spectrum'
        self.default_parameters = self.get_default_tokamak_parameters()

        self.kb   = 1.3806E-23 # m^2kg/s^2K
        self.k0   = 1.6022E-12 # erg/ev
        self.e0   = 4.8032E-10 # elementary charge (statcoulombs)
        self.e00  = 1.6020E-19 # elementary charge (C)
        self.c0   = 2.9979E+10 # speed of light (cm/sec)
        self.me   = 9.1093E-28 # electron mass (g)
        self.mee   = 9.1093E-31 # electron mass (kg)
        self.mp   = 1.6726E-24 # proton mass (g)
        self.mpp  = 1.6726E-27 # proton mass (kg)
        self.pi   = np.pi

    def write_input_file(self, params: dict, run_dir: str):
        # give some parameters write to a new input file!
        # TODO: write a standard input file based on somthing?
        print('Writing to', run_dir)
        if os.path.exists(run_dir):
            input_fpath = os.path.join(run_dir, 'input.tglf')
            subprocess.run(['touch', f'{input_fpath}'])
        else:
            raise FileNotFoundError(f'Couldnt find {run_dir}')

        run_parameters: dict = deepcopy(self.default_parameters)
        
        for param_name, val in params.items():
            if param_name not in run_parameters.keys(): 
                raise ValueError(f'{param_name} not included in the variables changable, please consult')
            run_parameters[param_name] = val 
        
        tglf_params = self.get_tglf_inputs_from_tokamak_parameters(run_parameters)
        
        with open(input_fpath, 'w') as file:
            for param_name, val in tglf_params.items(): 
                file.write(f'{param_name}={val}\n')

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

    def get_default_tokamak_parameters(self, ) -> dict: 
        geometric_params = {
            'ELON':  1.58,    # Elongation of a magnetic surface [-]
            'TRIA':  0.22,    # Triangularity of a magnetic surface [-]
            'A0':    0.5,     # mid plane major radius last closed flux surface [m] <- Not included in ASTRA input since it is last value of AMETR profile
            'BTOR':  2.5,     # toroidal field strength [T] 
            # need equilibrium file for below
            'RTOR':  1.6,     # major radius on toroidal axis [m], includes shafranov shift
            'RHO':   0.03,    # normalized flux coordinate (x-axis) [m]
            'VRS':   1.0,     # Volume gradient on intermediate grid at time t [m^2]
            'AMETR': 0.25,    # Magnetic surface radius in a mid-plane [m]
            # 'SHIF':  0.0,     # Shafranov shift of a magnetic surface 
            'G11':   1.0,     # <\grad(rho)^2>VRS [m^2]
            'MU':    0.17,    # inverse safety factor 1 / q? 
        }

        # these are not included in the ASTRA stuff but necessary
        gradient_params = {
            # "drhodrmin":   1.0,
            "drmin":  0.01,
            "dqdrmin" : 0.01, # used in q_prime and s
            "drmaj":  1.0,
            "dptotdrmin":  0.1,
            "drmajdrmin": 0.1, 
            'dq':     1.0,
            'dtriandrmin': 0.1, 
            'delondrmin':  0.1,
            'dvperdrmin':  0.1, # in principle is the same as ER
            'dv_rdrmin':        1.0, # line 349, difference in VPAR
            # normalized density gradients
            'dnedrmin':    -3.885974E-3, # E16, 
            'dtedrmin':    1.2641052E-3, 
            'dnidrmin':    -3.885974E-3,# E16, 
            'dtidrmin':    1.2641052E-3, 
            'dzti1drmin':  0.0,
            'dzti2drmin':  0.0, # impurity temperature gradient
            'dzti3drmin':  0.0,
            'dzni1drmin':  0.0,
            'dzni2drmin':  0.0,
            'dzni3drmin':  0.0,
        }

        # Main Plasma Parameters
        main_plasma_params = {
            # 'NE':  5.5446718,# E19,     # density electron
            'TE':  1.5961982,     # temperature electron [keV, as it will be multiplied by 1000] [1-]
            'NI':  5.5446718, # E19,     # density main ion species
            'TI':  1.5961982,     # temperature main ion species [keV, as it will be multiplied by 1000]
            'AMJ': 1.0,           # Main ion mass 
            'ZMJ': 1.0,           # Main ion charge
            'ZEF': 5.0,           # Integral by electron number (density * charge) -> profile
        }

        # Impurities
        impurity_params = {
            'NIZ1': 1E-9,   # density impurity species 1
            'NIZ2': 1E-9,   # density impurity species 2
            'NIZ3': 1E-9,   # density impurity species 3
            'ZIM1': 2.0,   # Charge (Ionization) of impurity 1 
            'ZIM2': 3.0,   # Charge (Ionization) of impurity 2 
            'ZIM3': 4.0,   # Charge (Ionization) of impurity 3
            "AIM1": 3.0,   # atomic mass of impurity 1 
            "AIM2": 3.0,   # atomic mass of impurity 2 
            "AIM3": 3.0,   # atomic mass of impurity 3
            "TIZ1": 1E-9,   # Temperature of impurity 1 [eV]
            "TIZ2": 1E-9,   # Temperature of impurity 1 [eV]
            "TIZ3": 1E-9,   # Temperature of impurity 1 [eV]
        }

        # Auxiliary Plasma Parameters

        # can condense P into just one total pressure
        auxiliary_plasma_params = {
            'VPOL': 0.0,   # Velocity of ions in poloidal direction
            'VTOR': 0.0,   # Velocity of ions in toroidal direction
            'ER': 0.0,     # radial electric field [V/m]
            'PFAST': 0.0,  # Pressure of fast ions [10^19 keV / m^3]
            'PBLON': 0.0,  # longitudinal Pressure of fast ions [10^19 keV / m^3]
            'PBPER': 0.0,  # Perpendicular pressure of fast ions [10^19 keV / m^3]
        }

        # Downstream 
        additional_params = {
            'SHEAR': 0.0,  # ??? (Current profile shearing)
        }

        params = {**additional_params, **auxiliary_plasma_params, **impurity_params, **main_plasma_params, **geometric_params, **gradient_params}

        return params 

    def get_astra_default_static_parameters(self, ) -> dict: 
        # TODO: static parameters
        static_parameters = {
            "XWELL_SA": 0.0,
            "THETA0_SA": 0.0,
            "S_ZETA_LOC": 0.0, 
            "ZETA_LOC": 0.0, 
            "DZMAJDX_LOC": 0.0,
            "DRMINDX_LOC": 1.0,
            "ZMAJ_LOC": 0.0,
            "B_MODEL_SA": 1, 
            "FT_MODEL_SA": 1, 
            "VPAR_SHEAR_MODEL": 1, 
            "VPAR_MODEL": 0, 
            "ETG_FACTOR": 1.25,
            "DEBYE_FACTOR": 1.0, 
            "XNU_FACTOR": 1.0, 
            "ALPHA_QUENCH": 0.0, 
            "ALPHA_MACH": 0.0, 
            "ALPHA_P": 1.0, 
            "ALPHA_E": 1.0, 
            "KX0_LOC": 0.0, 
            "UNITS": "CGYRO",
            "NKY": 19, 
            "NBASIS_MAX": 6, 
            "NMODES": 7,
            "USE_MHD_RULE": ".False.",
            "USE_BPER": ".True."
        }

        # TODO: loose params 

        variable_parameters = {
            "SAT_RULE": 2,
            "KYGRID_MODEL": 4, 
            "XNU_MODEL": 3,  
            "ALPHA_ZF": 1,
            # "WDIA_TRAP": 1,        
        }

        return {**variable_parameters, **static_parameters}


    def get_tglf_inputs_from_tokamak_parameters(self, t_params: dict) -> dict:
        Rmaj = t_params['RTOR']  # +  t_params['drmajdrmin'] # could add shafranov shift here!
        q = 1.0/t_params['MU']
        a0 = t_params['A0'] # minor radius of LCFS 
        r = t_params['AMETR']
        dr = t_params['AMETR']/a0
        main_ion_mass = t_params['AMJ']*self.mp

        # GYRO CONVENTIONS
        TE_GYRO = 1E3*t_params['TE']

        # NOTE: we do quasi-neutrality by not passing NE directly but by computing it
        # NE_GYRO = 1E13*t_params['NE']
        # total_charge_density = sum([species_parameters[f'ZS_{i}']*species_parameters[f'AS_{i}'] for i in [1, 2, 3, 4, 5]])
        electron_density = sum([t_params[f'ZIM{i}']*t_params[f'NIZ{i}'] for i in [1,2,3]] + [t_params['NI']*t_params['ZMJ']]) 
        electron_density_gradient = sum([t_params[f'dzni{i}drmin']*t_params[f'ZIM{i}']*t_params[f'NIZ{i}'] for i in [1,2,3]] + [t_params['dnidrmin']*t_params['NI']*t_params['ZMJ']]) 
        total_charge = (1.0 / electron_density) * (sum([(t_params[f'ZIM{i}']**2)*t_params[f'NIZ{i}'] for i in [1,2,3]] + [t_params['NI']*(t_params['ZMJ'])**2]) )
        NE_GYRO =  1E13*(electron_density)
        BUNIT = 1E4*t_params['BTOR'] # *(t_params['drhodrmin'])*t_params['RHO']/t_params['AMETR'] # Miller geometry magnetic field unit
        # above reasoning; rho,r ~unity for small r/rho, whereas for large rho, a, this can be > 1, so essentially just scan BT accordingly


        # some reused quantities
        # cuts corners using TE, should maybe use (TI_GYRO + TE_GYRO) instead but...
        ion_thermal_velocity = np.sqrt(self.k0*TE_GYRO / main_ion_mass) # cm / s
        ion_gyrofrequency = self.e0*BUNIT / (main_ion_mass*self.c0) # 1/sec
        ion_gyroradius = ion_thermal_velocity / ion_gyrofrequency # cm
        lnlamda = 24.0 - 0.5*np.log(NE_GYRO) + np.log(TE_GYRO) # collision term? 
        electron_collision_time = 3.44E5 * (TE_GYRO)**(1.5) / (NE_GYRO*lnlamda) # seconds

        BPOLZ = t_params['BTOR']*r / (q*Rmaj)
        BMOD = np.sqrt(t_params['BTOR']**2 + BPOLZ**2)
        VPAR = t_params['VTOR']*t_params['BTOR'] / BMOD + t_params['VPOL']*BPOLZ/BMOD
        
        geometry_parameters = {
            # geometry parameters
            "Q_LOC": q, 
            "Q_PRIME_LOC": (q*(a0**2)/r)*t_params['dqdrmin'], # q / (t_params['AMETR'] / a0)*(t_params['dq']/dr),
            "P_PRIME_LOC": (self.k0/BUNIT**2)*(q*a0**2/r)*t_params['dptotdrmin'], # (self.k0/BUNIT**2)*(q/(r / a0))*(t_params['dptot']/dr),
            "S_DELTA_LOC": r*t_params['dtriandrmin'],
            "S_KAPPA_LOC": r*t_params['delondrmin'],
            "DELTA_LOC":   t_params['TRIA'],
            "KAPPA_LOC":   t_params['ELON'],
            "DRMAJDX_LOC": t_params['drmajdrmin'], 
            "RMAJ_LOC":    Rmaj / a0,
            "RMIN_LOC":    r/a0,
        }

        shear_parameters = {
            # Shearing parameters
            'ALPHA_SA': -(8.0*self.pi*self.k0/BUNIT**2)*(q**2)*(Rmaj*t_params['dptotdrmin']),
            'SHAT_SA': (r / q)*t_params['dqdrmin'], # (r / q)*(t_params['dq']/t_params['drmin']),
            'Q_SA': q, 
            'RMAJ_SA': Rmaj / a0, 
            "RMIN_SA": r / a0,
        }     

        plasma_paramters = {
            # plasma parameters
            "ZEFF": total_charge, # t_params['ZEF'],
            "XNUE": 0.75*np.sqrt(self.pi)*a0 / (ion_thermal_velocity*electron_collision_time),
            "BETAE": (8.0*self.pi)*(self.k0*NE_GYRO*TE_GYRO) / BUNIT**2,
            "DEBYE": np.sqrt(self.k0*TE_GYRO / (4.0*self.pi*NE_GYRO*self.e0**2)) / ion_gyroradius,
            "VEXB_SHEAR": -1E2*(a0*r/q)*t_params['dvperdrmin'] / ion_thermal_velocity, # Waltz-miller definition 
            # TODO: TGLF MAN PAGE SAYS BELOW IS NOT IN USE; SEE VPAR  
            "VEXB": 1E2*(-t_params['ER']/BMOD)/ion_thermal_velocity, # TODO: check if the 1E2 is needed?
        }

        species_parameters = {
            # electron
            "RLNS_1": electron_density_gradient * (a0 / electron_density),
            "RLTS_1": t_params['dtedrmin'] * (a0 / t_params['TE']),
            "AS_1": 1.0,    # electron density is reference
            "TAUS_1": 1.0,  # electron temperature is reference
            "ZS_1": -1.0, 
            "MASS_1": 5.4447E-4 / t_params['AMJ'],
            "VPAR_1": 1E2*VPAR / ion_thermal_velocity, # same as VPAR_2
            "VPAR_SHEAR_1": -1E2*Rmaj*a0*t_params['dv_rdrmin'] / (ion_thermal_velocity), # same as VPAR_SHEAR 2


            # main ion species
            "RLNS_2": t_params['dnidrmin'] * (a0 / t_params['NI']),
            "RLTS_2": t_params['dtidrmin'] * (a0/ t_params['TI']),
            "AS_2": t_params['NI'] / electron_density,
            "TAUS_2": t_params['TI'] / t_params['TE'],
            "ZS_2": t_params['ZMJ'],
            "MASS_2": 1.0, 
            "VPAR_2": 1E2*VPAR / ion_thermal_velocity,
            "VPAR_SHEAR_2": -1E2*Rmaj*a0*t_params['dv_rdrmin'] / (ion_thermal_velocity), # same as VPAR_SHEAR 2
            
            # impurities
            "RLNS_3": t_params['dzni1drmin'] * (a0 / t_params['NIZ1']),
            "RLTS_3": t_params['dzti1drmin'] * (a0 / t_params['TIZ1']),
            "AS_3": t_params['NIZ1'] / electron_density,
            "TAUS_3": t_params['TIZ1'] / t_params['TE'],
            "ZS_3": t_params['ZIM1'],
            "MASS_3": t_params['AIM1'] / t_params['AMJ'],
            "VPAR_3": 1E2*VPAR / ion_thermal_velocity, # same as VPAR_2
            "VPAR_SHEAR_3": -1E2*Rmaj*a0*t_params['dv_rdrmin'] / (ion_thermal_velocity), # same as VPAR_SHEAR 2
            
            "RLNS_4": t_params['dzni2drmin'] * (a0 / t_params['NIZ2']),
            "RLTS_4": t_params['dzti2drmin'] * (a0 / t_params['TIZ2']),
            "AS_4": t_params['NIZ2'] / electron_density,
            "TAUS_4": t_params['TIZ2'] / t_params['TE'],
            "ZS_4": t_params['ZIM2'],
            "MASS_4": t_params['AIM2'] / t_params['AMJ'],
            "VPAR_4":  1E2*VPAR / ion_thermal_velocity, # same as VPAR_2
            "VPAR_SHEAR_4": -1E2*Rmaj*a0*t_params['dv_rdrmin'] / (ion_thermal_velocity), # same as VPAR_SHEAR 2

            "RLNS_5": t_params['dzni3drmin'] * (a0 / t_params['NIZ3']),
            "RLTS_5": t_params['dzti3drmin'] * (a0 / t_params['TIZ3']),
            "AS_5": t_params['NIZ3'] / electron_density,
            "TAUS_5": t_params['TIZ3'] / t_params['TE'],
            "ZS_5": t_params['ZIM3'],
            "MASS_5": t_params['AIM3'] / t_params['AMJ'],
            "VPAR_5":  1E2*VPAR / ion_thermal_velocity, # same as VPAR_2
            "VPAR_SHEAR_5": -1E2*Rmaj*a0*t_params['dv_rdrmin'] / (ion_thermal_velocity), # same as VPAR_SHEAR 2
        }


        # quasi-neutrality for density and density gradient 
        # total_charge_density = sum([species_parameters[f'ZS_{i}']*species_parameters[f'AS_{i}'] for i in [1, 2, 3, 4, 5]])
        # total_charge_density_gradient = sum([species_parameters[f'RLNS_{i}']*species_parameters[f'ZS_{i}']*species_parameters[f'AS_{i}'] for i in [1, 2, 3, 4, 5]])
        # species_parameters['AS_2'] = -(1.0 /species_parameters['ZS_2']) *(total_charge_density  - species_parameters['AS_2']*species_parameters['ZS_2'])
        # species_parameters['RLNS_2'] = -(1.0 / (species_parameters['AS_2']*species_parameters['ZS_2']))*(total_charge_density_gradient - species_parameters['AS_2']*species_parameters['ZS_2']*species_parameters['RLNS_2'])
        # NOTE: we do quasi-neutrality by not passing NE directly but by computing it


        static_parameters = self.get_astra_default_static_parameters()
        return {**shear_parameters, **geometry_parameters, **plasma_paramters, **species_parameters, **static_parameters}
        
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