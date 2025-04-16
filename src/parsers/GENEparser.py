import os, sys
from .base import Parser
import f90nml
import numpy as np
import re
from dask.distributed import print

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
        self.base_params_nml_string = '''
            &parallelization
                n_parallel_sims = 1
                n_procs_sim = 128
                n_procs_s = -1
                n_procs_z = -1
                n_procs_w = -1
                n_procs_v = -1
                n_procs_x = -1
                n_procs_y = -1
            /

            &box
                n_spec = 2
                nx0 = 18
                nky0 = 1
                nz0 = 36
                nv0 = 32
                nw0 = 16
                kymin = 0.1
                lv = 3.1
                lw = 11
                x0 = 0.9
                kx_center = 0.0
                adapt_ly = .true.
            /

            &in_out
                diagdir = '/scratch/project_462000451/gene_out/gene_auto_97781/'
                read_checkpoint = .false.
                istep_nrg = 100
                istep_field = 100
                istep_mom = 2000
                istep_energy = 100
                istep_omega = 100
                istep_vsp = 5000
                istep_schpt = 5000
                istep_srcmom = 2000
                iterdb_file = '/project/project_462000451/jet_97781_data/iterdb_jet_97781'
            /

            &general
                perf_vec = 2, 1, 2, 2, 1, 2, 2, 1, 2
                nblocks = 16
                f_version = .false.
                nonlinear = .false.
                x_local = .true.
                arakawa_zv = .false.
                comp_type = 'IV'
                calc_dt = .true.
                timelim = 7200
                simtimelim = 7200
                underflow_limit = 1e-15
                collision_op = 'landau'
                coll_cons_model = 'self_adj'
                coll = -1
                zeff = 1
                beta = -1
                debye2 = -1
                hyp_z = -1
                init_cond = 'ppjrn'
            /

            &geometry
                edge_opt = 1.0
                magn_geometry = 'tracer_efit'
                geomdir = ''
                geomfile = '/project/project_462000451/jet_97781_data/jet97781.eqdsk'
                rhostar = -1
                mag_prof = .true.
                sign_ip_cw = 1
                sign_bt_cw = 1
            /

            &species
                name = 'Electrons'
                mass = 0.0002725
                charge = -1
                prof_type = -2
            /

            &species
                name = 'Ions'
                mass = 1.0
                charge = 1
                prof_type = -2
            /

            &units
                tref = -1
                nref = -1
                bref = -1
                lref = -1
                mref = 2
                omegatorref = -1
            /
            '''
        self.parameter_nml_map={
            'kymin':('box','kymin'),
            'omn1':('_grp_species_0','omn'),
            'omn2':('_grp_species_1','omn'),
            'omt1':('_grp_species_0','omt'),
            'omt2':('_grp_species_1','omt'),
            'coll':('general','coll')
        }
    
    def write_input_file(self, params: dict, run_dir: str, base_parameters_file_path: str):
        if base_parameters_file_path != None:
            namelist = f90nml.read(base_parameters_file_path)
            namelist_string = str(namelist)
        
        elif self.base_params_nml_string != None:
            namelist = f90nml.reads(self.base_params_nml_string)
            namelist_string = self.base_params_nml_string
        
        #populate params: dict with all omn's required. Since each should be identical
        if 'omn' in params:
            print('species omn is being handeled by making species 1 and 2 with the same omn')
            for i in range(namelist_string.count('&species')):
                i+=1
                params[f'omn{i}'] = params['omn']
            params.pop('omn')
        
        for p in params.keys():
            if not p in self.parameter_nml_map.keys():
                raise ValueError(f'THERE IS NO ENTRY FOR PARAMETER {p} IN THE parameter_nml_map:\n{self.parameter_nml_map}\n PLEASE ADD IT IN THE GENEparser')
            
        parameters_path = os.path.join(run_dir, 'parameters')
        
        patch = {}
        for key, value in params.items():
            group, var = self.parameter_nml_map[key]
            if group in patch.keys():
                patch[group][var] = value
            else:
                patch[group] = {var:value}
            # patch = {group: {var: value}}
        patch['in_out'] = {'diagdir':run_dir}
        
        namelist.patch(patch)
        # print('check 0 omt',namelist['_grp_species_1']['omt'], params[('_grp_species_1','omt')])
        # print('check 1 omt',namelist['_grp_species_2']['omt'], params[('_grp_species_2','omt')])
        # print('check 0 omn',namelist['_grp_species_1']['omn'], params[('_grp_species_1','omn')])
        # print('check 1 omn',namelist['_grp_species_2']['omn'], params[('_grp_species_2','omn')])
        # print(namelist['_grp_species_1'])        
        f90nml.write(namelist, parameters_path)
    
    def calculate_kperp(self, run_dir, suffix='.dat'):
        current_dir = os.getcwd()
        os.chdir(run_dir)
        pars = init_read_parameters_file(suffix)
        field_path = GFF(run_dir).find_files('field')[0]
        field = GF(field_path)
        field_dict = field.field_filepath_to_dict(time_criteria='last')
        zgrid = field_dict['zgrid']
        apar = np.abs(field_dict['field_apar'][-1])
        phi = np.abs(field_dict['field_phi'][-1])
        
        geom_type, geom_pars, geom_coeff = init_read_geometry_file(suffix,pars)
        kperp, omd_curv, omd_gradB = calc_kperp_omd(geom_type,geom_coeff,pars,False,False)
        
        avg_kperp_squared_phi = np.sum((phi/np.sum(phi)) * kperp**2)
        avg_kperp_squared_A = np.sum((apar/np.sum(apar)) * kperp**2)
        
        os.chdir(current_dir)
        
        return avg_kperp_squared_phi, avg_kperp_squared_A
    
    def calculate_fingerprints(self, run_dir, suffix='.dat'):
        current_dir = os.getcwd()
        os.chdir(run_dir)
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
    
    def read_omega(self, run_dir, suffix='.dat'):
        omega_path = os.path.join(run_dir,'omega'+suffix)
        with open(omega_path, 'r') as file:
            line = file.read()
            vars = line.split(' ')
            vars = [v for v in vars if ' ' not in v]
            vars = [float(v) for v in vars if v != '']
            ky, growthrate, frequency = vars
        return ky, growthrate, frequency
    
    def read_output(self, run_dir, suffix='.dat', get_required_files=False):
        if get_required_files:
            return ['field','omega','nrg','parameters']
        avg_kperp_squared_phi, avg_kperp_squared_A = self.calculate_kperp(run_dir, suffix)
        ky, growthrate, frequency = self.read_omega(run_dir, suffix)
        mixing_length_phi = growthrate / avg_kperp_squared_phi
        mixing_length_A = growthrate / avg_kperp_squared_A
        fingerprints = self.calculate_fingerprints(run_dir, suffix)
        return mixing_length_phi, mixing_length_A, *fingerprints, growthrate, frequency

    def print_default_nml_string(self, base_parameters_file_path):
        #to prevent each worker needing to read the base parameters file
        # you can print the default namelist string and paste it into the parser.
        nml = f90nml.read(base_parameters_file_path)
        print("COPY AND PASTE THE STRING BELOW INTO, GENEparser --> __init__ --> self.base_params_nml_string, FOR OPTMAL PERFORMANCE")
        print(str(nml))
        return str(nml)
    
    # Scan Parser Functions_________________________
    def latest_scanfiles_dir(self, run_dir):
        dirs = os.listdir(run_dir)
        scanfiles_number = [re.findall('scanfiles[0-9]{4}',sc_dir) for sc_dir in dirs]
        scanfiles_number = [item for sublist in scanfiles_number for item in sublist]
        scanfiles_number = [re.findall('[0-9]{4}',sn) for sn in scanfiles_number]
        scanfiles_number = [item for sublist in scanfiles_number for item in sublist]
        latest_scan_dir = os.path.join(run_dir,f'scanfiles{np.sort(np.array(scanfiles_number))[-1]}')
        return latest_scan_dir
    

    def check_status(self,scanfiles_dir):
        status_path = os.path.join(scanfiles_dir, 'in_par','gene_status')
        with open(status_path, 'r') as status_file:
            status = status_file.read()#.decode('utf8')
            return status
    
    def write_scan_file(self, run_dir, params, base_parameters_file_path, n_jobs=1):            
        parameters_path = os.path.join(run_dir,'parameters')
        num_params = len(params)
        def merge_dicts(dict_list):
            keys = dict_list[0].keys()
            return {key: [d[key] for d in dict_list] for key in keys}
        params = merge_dicts(params)
        
        if base_parameters_file_path != None:
            namelist = f90nml.read(base_parameters_file_path)
            namelist_string = str(namelist)
        
        elif self.base_params_nml_string != None:
            namelist = f90nml.reads(self.base_params_nml_string)
            namelist_string = self.base_params_nml_string
    
        #populate params: dict with all omn's required. Since each should be identical
        if 'omn' in params:
            print('species omn is being handeled by making species 1 and 2 with the same omn')
            for i in range(namelist_string.count('&species')):
                i+=1
                params[f'omn{i}'] = params['omn']
            params.pop('omn')
                
        def find_nth_occurrence(string, sub_string, n):
            start_index = string.find(sub_string)
            while start_index >= 0 and n > 1:
                start_index = string.find(sub_string, start_index + 1)
                n -= 1
            return start_index

        # finds the string location at the end of the line for a variable, just before \n
        def var_end_loc(namelist_string: str, param_key):
            group_name, var_name = self.parameter_nml_map[param_key]
            group_ordinal = 0 #0 is the 1st
            if len(group_name.split('_'))>1:
                # print('MORE THAN ONE GROUP OF SAME NAME')
                _, _, group_name, group_ordinal = group_name.split('_')
                group_ordinal = int(group_ordinal)+1

            # print('GROUP NAME',group_name,'VAR NAME',var_name, 'ORDIANL',group_ordinal)

            group_start = find_nth_occurrence(namelist_string, group_name, group_ordinal)
            group_end = group_start+namelist_string[group_start:].find(f'/')
            # print('GROUP',namelist_string[group_start:group_end])

            var_start = group_start+namelist_string[group_start:group_end].find(var_name+' ') #space ensures it is not apart of another name. Only works if there is a space after every variable
            var_end = var_start+namelist_string[var_start:group_end].find("\n")
            # print('VARLOC',var_start,var_end)
            # print('VAR',namelist_string[var_start:var_end])
            # print('START',namelist_string[var_start],'END',namelist_string[var_end])
            return var_end

        def make_scanlist(values):
        # Making scanlist
            scanlist = f'      !scanlist: {values[0]}'
            for v in values[1:]:
                scanlist += f', {v}'
            return scanlist
        #----------------------
        
        #determines the ordinal position of the scanned paramters for var_name
        def var_ordinal(param_key):
            group, var_name = self.parameter_nml_map[param_key]
            group_split = group.split('_')
            if len(group_split)>1:
                group_name = group_split[-2]
                group_ord = int(group_split[-1])+1
            else:
                group_name = group
                group_ord = 1

            gloc = find_nth_occurrence(namelist_string, group_name, group_ord)
            vloc = gloc + namelist_string[gloc:].find(var_name)
            var_ordinal = namelist_string[:vloc].count('=')+1
            return var_ordinal
        
        #check which parameter is the first to be scanned and make is a scanlist
        ordinals = {k:var_ordinal(k) for k in params.keys()}
        first_param = None
        first_param_ord = np.inf
        for k,ord in ordinals.items():
            if ord < first_param_ord: 
                first_param_ord = ord
                first_param = k
        
        def make_scanwith(values):
            scanwith = f'       !scanwith: 0, {values[0]}'
            for v in values[1:]:
                scanwith += f', {v}'
            return scanwith

        scanwith = {k:make_scanwith(values) for k,values in params.items()}
        # Add scanwith to each variable with and scanlist to the first one
        for param_key in list(params.keys()):
            if param_key == first_param:
                var_end = var_end_loc(namelist_string,param_key)
                namelist_string = namelist_string[:var_end] + make_scanlist(params[param_key]) + namelist_string[var_end:]
            else:
                var_end = var_end_loc(namelist_string,param_key)
                namelist_string = namelist_string[:var_end] + scanwith[param_key] + namelist_string[var_end:]
        # placing in the remote save directory
        lines = namelist_string.split('\n')
        for line, i in zip(lines, np.arange(len(lines))):
            if 'diagdir' in line: lines[i] = f"    diagdir = '{run_dir}'" 
            if 'n_parallel_sims' in line: lines[i] = f"    n_parallel_sims = {n_jobs}" 
        namelist_string = '\n'.join(lines)
        #Writing the final namelist stirng to file. This is the scan parameters file.
                # checking run dir exists and making Path for scan file
        print('Writing to', parameters_path)
        with open(parameters_path, 'w') as file:
            file.write(namelist_string)  
        return namelist_string
    
    def write_sbatch(self, run_dir, sbatch_string, wallseconds):
        sbatch_path = os.path.join(run_dir,'submit.cmd')
        continue_path = os.path.join(run_dir, 'continue.cmd')
        print('WRITE SBATCH')
        sbatch_lines = sbatch_string.splitlines()
        wall_clock_limit = sec_to_time_format(wallseconds)
        wall_loc = 0
        for i in range(len(sbatch_lines)):
            if '#SBATCH -t' in sbatch_lines[i]: 
                wall_loc = i
                break
        sbatch_lines[wall_loc] = f"#SBATCH -t {wall_clock_limit}  ## wallclock limit, dd-hh:mm:ss\n"

        sbatch = "".join(sbatch_lines)

        with open(sbatch_path, 'w') as sbatch_file:
            sbatch_file.write(sbatch)

        # Make contiue scan script
        for i, line in enumerate(sbatch_lines):
            if "./scanscript" in line:
                sbatch_lines[i] = line.replace("./scanscript", "./scanscript --continue_scan")

        continue_str = "".join(sbatch_lines)
        with open(sbatch_continue_path, "w") as continue_file:
            continue_file.write(continue_str)
        return sbatch

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

    