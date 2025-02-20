import numpy as np

# from TPED.projects.profile_tools.utils.read_GENE_prof import GENEProfileReader
# from TPED.projects.profile_tools.utils.read_pfile import PFileReader
# from TPED.projects.utils.interpolation_tools import interp
from TPED.projects.utils.finite_differences import fd_d1_o4_uneven



#mode = 'eta': increase omt(e,i) by alpha, decrease omn to keep pressure fixed
#mode = 'etaTe': increase omte by alpha, decrease omti 
#mode = 'coll': vary collisionality at fixed pressure by decreasing/increasing temperature/density by alpha
#mode = 'TiTe': increase Te by alpha, decrease Ti by alpha
#mode = 'omnz': increase omnz (i.e. impurity density gradient) by alpha and decrease omni 
#mode = 'omte': increase omte without compensation
#mode = 'omti': increase omti without compensation
#mode = 'omt': modify both omti and omte without compensation
#mode = 'omne': increase omne without compensation
#mode = 'broaden_profiles': broaden ne by factor alpha


class PlasmaProfileModifier:
    def __init__(self, input_profile_xarray):
        self.profile_xarray = input_profile_xarray.copy(deep=True)
        # self.center_point = 0    
        # self.boundary_point = 0
    

    def eta_mode(self, scaling_loc, alpha):
        prof_xarray = self.profile_xarray

        ne = prof_xarray['ne']
        Te = prof_xarray['Te']

        ni = prof_xarray['ni']
        Ti = prof_xarray['Ti']

        # Find the index where 'rho_tor' is closest to rhotMidPed
        rho_tor = prof_xarray['rho_tor'].values
        scaling_index = np.argmin(abs(rho_tor - scaling_loc))

        # Extract 'Te' and 'Ti' at the midPedIndex using the index coordinate directly
        Te_scaling_pos = prof_xarray['Te'].isel(index=scaling_index).item()  # Get scalar value
        Ti_scaling_pos = prof_xarray['Ti'].isel(index=scaling_index).item()  # Get scalar value

        Te_alpha = Te_scaling_pos*np.power(Te/Te_scaling_pos, alpha)
        Ti_alpha = Ti_scaling_pos*np.power(Ti/Ti_scaling_pos, alpha)

        prof_xarray['Te'] = Te_alpha
        prof_xarray['Ti'] = Ti_alpha
        
        if ('Tz' in prof_xarray):
            Tz = prof_xarray['Tz']
            Tz_scaling_pos = prof_xarray['Tz'].isel(index=scaling_index).item()
            Tz_alpha = Tz_scaling_pos*np.power(Tz/Tz_scaling_pos, alpha)
            prof_xarray['Tz'] = Tz_alpha

        Ptot_orig = self._calc_pressure()
        Ptot_alpha = self._calc_pressure(prof_xarray)

        ne_alpha = ne*(Ptot_orig/Ptot_alpha)
        ni_alpha = ni*(Ptot_orig/Ptot_alpha)
        prof_xarray['ne'] = ne_alpha
        prof_xarray['ni'] = ni_alpha

        if ('nz' in prof_xarray):
            nz = prof_xarray['nz']
            prof_xarray['nz'] = nz*(Ptot_orig/Ptot_alpha)
        
        etae_original = ne/Te*fd_d1_o4_uneven(Te,rho_tor)/fd_d1_o4_uneven(ne,rho_tor)
        etae_alpha = ne_alpha/Te_alpha*fd_d1_o4_uneven(ne_alpha,rho_tor)/fd_d1_o4_uneven(ne_alpha,rho_tor)

        prof_xarray['etae'] = etae_original
        prof_xarray['etae_alpha'] = etae_alpha

        return prof_xarray




    def broaden_profiles(self, profile_list, alpha):
        rhotTopPed =0.92



        # rhot = profile_dict['ne']['rho_tor']
        # ne = profile_dict['ne']['data']
        # Te = profile_dict['Te']['data']

        # ni = profile_dict['ni']['data']
        # Ti = profile_dict['Ti']['data']

        # nz = profile_dict['nz']['data']
        # Tz = profile_dict['Tz']['data']




        # # Code to broaden ne by factor alpha
        # ipedtop = np.argmin(abs(rhot-rhotTopPed))
        # nind_pedtop = len(rhot) - ipedtop
        # ped_domain = 1-rhotTopPed
        # ped_domain_broad = alpha*ped_domain
        # rhotTopPed_broad = 1-ped_domain_broad
        # xtemp_ped = np.linspace(rhotTopPed_broad,1,nind_pedtop)
        # xtemp_core = np.linspace(0,rhotTopPed_broad,len(rhot)-nind_pedtop,endpoint=False)
        
        # rhot_new = np.concatenate([xtemp_core,xtemp_ped])
        # newNe = interp(rhot_new,ne,rhot)
        # newNi = interp(rhot_new,ni,rhot)
        # Z = float(input('Enter Z of impurity:\n'))
        
        # newNz = (newNe - newNi)/Z
        # newTe = interp(rhot_new,Te,rhot)
        # newTi = interp(rhot_new,Ti,rhot)
        # newTz = interp(rhot_new,Tz,rhot)
        # newPtot = newNe * newTe + newTi * newNi + newTz*newNz

        
        
        # return modified_profiles

    # def vary_collisionality(self, profiles, alpha):
    #     # Code to vary collisionality at fixed pressure by decreasing/increasing temperature/density by alpha
    #     return modified_profiles

    # def modify_eta(self, profiles, alpha):
    #     # Code to modify eta based on the given mode
    #     return modified_profiles

    # # Define other functions for each mode similarly



    def _calc_pressure(self, input_prof_xarray = None):

        if input_prof_xarray is None:
            input_prof_xarray = self.profile_xarray

        # Check for variables and extract if they exist
        Tz = input_prof_xarray['Tz'] if 'Tz' in input_prof_xarray.variables else None
        nz = input_prof_xarray['nz'] if 'nz' in input_prof_xarray.variables else 0

        # Assuming Te, Ti, ne, ni are always present
        Te = input_prof_xarray['Te']
        Ti = input_prof_xarray['Ti']
        ne = input_prof_xarray['ne']
        ni = input_prof_xarray['ni']

        # Compute total pressure based on available data
        if Tz is not None:
            # All variables are available
            Ptot = Te*ne + Ti*ni + Tz*nz
        elif nz > 0:
            # nz is available but Tz is not, adjust Ti term
            Ptot = Te*ne + Ti*(ni + nz)
        else:
            # Only Ti, Te, ni, ne are available
            Ptot = Te*ne + Ti*ni

        return Ptot



