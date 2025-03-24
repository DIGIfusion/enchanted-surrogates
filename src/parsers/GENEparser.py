import os, sys
from .base import Parser
import f90nml
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'submodules', 'IFS_scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'submodules'))

from parsers.submodules.IFS_scripts.geomWrapper import calc_kperp_omd, init_read_geometry_file
from parsers.submodules.IFS_scripts.parIOWrapper import init_read_parameters_file
from parsers.submodules.IFS_scripts.get_nrg import get_nrg0
# from GENE_ML.IFS_scripts.fieldHelper import fieldfile, field_xz

from parsers.submodules.TPED.projects.GENE_sim_reader.archive.ARCHIVE_GENE_field_data import GeneField as GF
from parsers.submodules.TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder as GFF


class GENEparser(Parser):
    def __init__(self):
        None

    def write_input_file(self, params: dict, run_dir: str, base_params_file_path: str):
        parameters_path = os.path.join(run_dir, 'parameters')
        namelist = f90nml.read(base_params_file_path)
        namelist_string = str(namelist)
        # params example {('box','kymin'): 0.5}
        #populate params: dict with all omn's required. Since each should be identical
        if ('species','omn') in params:
            print('species omn is being handeled by making species 1 and 2 with the same omn')
            for i in range(namelist_string.count('&species')):
                params[(f'_grp_species_{i}','omn')] = params[('species','omn')]
            params.pop(('species','omn'))
        
        patch = {}
        for key, value in params.items():
            group, var = key
            patch[group] = {var:value}
            # patch = {group: {var: value}}
        namelist.patch(patch)
        
        f90nml.write(parameters_path, namelist)
    
    def calculate_kperp(self, out_dir):
        current_dir = os.getcwd()
        os.chdir(out_dir)
        pars = init_read_parameters_file('.dat')
        field_path = GFF(out_dir).find_files('field')[0]
        field = GF(field_path)
        field_dict = field.field_filepath_to_dict(time_criteria='last')
        zgrid = field_dict['zgrid']
        apar = np.abs(field_dict['field_apar'][-1])
        phi = np.abs(field_dict['field_phi'][-1])
        
        geom_type, geom_pars, geom_coeff = init_read_geometry_file('.dat',pars)
        kperp, omd_curv, omd_gradB = calc_kperp_omd(geom_type,geom_coeff,pars,False,False)
        
        avg_kperp_squared_phi = np.sum((phi/np.sum(phi)) * kperp**2)
        avg_kperp_squared_A = np.sum((apar/np.sum(apar)) * kperp**2)
        
        os.chdir(current_dir)
        
        return avg_kperp_squared_phi, avg_kperp_squared_A
    
    def calculate_fingerprints(self, out_dir):
        current_dir = os.getcwd()
        os.chdir(out_dir)
        pars = init_read_parameters_file('.dat')
        time, nrg1, nrg2 = get_nrg0('.dat', nspec=2)
        
        fluxes = [read_Gamma_Q(time, nrg1, print_val=False, setTime=-1), read_Gamma_Q(time, nrg2, print_val=False, setTime=-1)]        
        
        D_all = []
        chi_all = []
        for i, fluxes_ in enumerate(fluxes):
            species_index = i+1
            Lref = pars['Lref']
            T = pars[f'temp{species_index}'] * pars['Tref']
            n = pars['nref'] * pars[f'dens{species_index}']
            omn = pars[f'omn{species_index}']
            grad_n = -(n/Lref) * omn
            # Gamma is particle flux, Q is heat flux.
            Gamma_es, Gamma_em, Q_es, Q_em = fluxes_
            Gamma = Gamma_em + Gamma_es
            Q_tot = Q_em + Q_es
            #Particle diffusivity
            D = - Gamma / grad_n 
            D_all.append(D)
            grad_T = pars[f'omt{species_index}'] * -(T/Lref)
            chi = -(Q_tot - (3/2)*T*Gamma)/(n * grad_T)
            chi_all.append(chi)
        #Requires electrons are the first species in the gene parameters file, 0 for electrons, 1 for ions
        fingerprints = (chi_all[1]/chi_all[0], D_all[0]/chi_all[0])
        os.chdir(current_dir)
        return fingerprints
    
    def read_omega(self, out_dir):
        omega_path = os.path.join(out_dir,'omega.dat')
        with open(omega_path, 'r') as file:
            line = file.read()
            vars = line.split(' ')
            vars = [v for v in vars if ' ' not in v]
            vars = [float(v) for v in vars if v != '']
            ky, growthrate, frequency = vars
        return ky, growthrate, frequency
    
    def read_output(self, out_dir):
        avg_kperp_squared_phi, avg_kperp_squared_A = self.calculate_kperp(out_dir)
        ky, growthrate, frequency = self.read_omega(out_dir)
        
        mixing_length_phi = growthrate / avg_kperp_squared_phi
        mixing_length_A = growthrate / avg_kperp_squared_A

        fingerprints = self.calculate_fingerprints(out_dir)
        
        return mixing_length_phi, mixing_length_A, fingerprints
def read_Gamma_Q(time,nrgs,print_val,setTime=-1):
     
    if (setTime == -1):
        this_nrg = nrgs[setTime,:]
        print ('Reading nrg file are at t = '+str(time[setTime]) )
    else:
        isetTime = np.argmin(abs(time-setTime))
        this_nrg = nrgs[isetTime]
        print ('Reading nrg file are at t = '+str(time[setTime]) )


    Gamma_es = this_nrg[4]
    Gamma_em = this_nrg[5]
    Qheat_es = this_nrg[6]
    Qheat_em = this_nrg[7]
    Pimom_es = this_nrg[8]
    Pimom_em = this_nrg[9]

    if print_val:
       #print "Gamma_es =", Gamma_es
       print ("Gamma_es = %12.4e" % Gamma_es)
       print ("Gamma_em = %12.4e" % Gamma_em)
       print ("Q_es = %12.4e" % Qheat_es)
       print ("Q_em = %12.4e" % Qheat_em)

    return Gamma_es, Gamma_em, Qheat_es, Qheat_em
        

