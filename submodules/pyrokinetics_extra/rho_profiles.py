"""
Reads in an iterdb file to get the profiles.
"""

import re
# from contextlib import redirect_stdout
from textwrap import dedent

import numpy as np
# from freeqdsk import peqdsk


from pyrokinetics.constants import deuterium_mass, electron_mass
from pyrokinetics.equilibrium import Equilibrium
from pyrokinetics.file_utils import FileReader
from pyrokinetics.species import Species
from pyrokinetics.typing import PathLike
from pyrokinetics.units import UnitSpline
from pyrokinetics.units import ureg as units
from pyrokinetics.units import PyroQuantity
from pyrokinetics.kinetics import Kinetics


# def ion_species_selector(nucleons, charge):
#     """
#     Returns ion species_name type from:

#     hydrogen deuterium, tritium, helium3, helium, other impurity.

#     Might need to update with more specific masses, such as 6.94 for Li-7, etc.
#     """
#     if nucleons == 1:
#         if charge.m == 1:
#             return "hydrogen"
#         else:
#             print(
#                 "You have a species_name with a single nucleon which is not a proton. Strange. Returning neutron for now. \n"
#             )
#             return "neutron"
#     if nucleons == 2 and charge.m == 1:
#         return "deuterium"
#     elif nucleons == 4 and charge.m == 2:
#         return "helium"
#     elif nucleons == 3:
#         if charge.m == 1:
#             return "tritium"
#         if charge.m == 2:
#             return "helium3"
#     else:
#         return "impurity"


# def np_to_T(n, p):
#     """
#     n is in m^{-3}, T is in eV, p is in Pascals.
#     Returns temperature in eV.
#     """
#     return np.divide(p, n).to("eV")

def species_dict_to_kinetics(species_dict:dict, equilibrium:Equilibrium) -> Kinetics:
    """
    dict: Holds the temp(rho) and density(rho) profiles for each species_name
    species_name:
        mass: kg
        charge: elementary charge
        temp_profile:
            rho:
            temp: eV 
        dens_profile:
            rho:
            dens: m^-3 
    """
    # Use the equilibrium to make the rho function
    rho_g = equilibrium["r_minor"].data.magnitude
    rho_g = rho_g / rho_g[-1] * units.lref_minor_radius
    psi_n_g = equilibrium["psi_n"].data
    rho_func = UnitSpline(psi_n_g, rho_g)
    psi_func = UnitSpline(rho_g, psi_n_g)
    
    result = {}
    for species_name in species_dict.keys():
        temp_rho = PyroQuantity(species_dict[species_name]['temp_profile']['rho'] * units.dimensionless)
        temp_psi = psi_func(temp_rho)
        temp = PyroQuantity(species_dict[species_name]['temp_profile']['temp'] * units.eV    )
        print(species_dict[species_name].keys())
        dens_rho = PyroQuantity(species_dict[species_name]['dens_profile']['rho'] * units.dimensionless)
        dens_psi = psi_func(dens_rho)
        dens = PyroQuantity(species_dict[species_name]['dens_profile']['dens'] * units.meter**-3)
        
        temp_func = UnitSpline(temp_psi, temp)
        dens_func = UnitSpline(dens_psi, dens)
        
        
        unit_charge_array = np.ones(len(psi_n_g))
        charge_func = UnitSpline(psi_n_g, species_dict[species_name]['charge'] * unit_charge_array * units.elementary_charge)
        
        result[species_name] = Species(
            # species_type=fast_species,
            charge=charge_func,
            mass=species_dict[species_name]['mass'],
            dens=dens_func,
            temp=temp_func,
            # omega0=omega_func,
            rho=rho_func,
        )
    
    return Kinetics(kinetics_type="pFile", **result), psi_func
    
        
        
    # # Interpolate on psi_n.
    # te_psi_n = profiles["te"]["psinorm"] * units.dimensionless
    # electron_temp_data = profiles["te"]["data"] * 1e3 * units.eV
    # electron_temp_func = UnitSpline(te_psi_n, electron_temp_data)

    # ne_psi_n = profiles["ne"]["psinorm"] * units.dimensionless
    # electron_dens_data = profiles["ne"]["data"] * 1e20 * units.meter**-3
    # electron_dens_func = UnitSpline(ne_psi_n, electron_dens_data)


    # unit_charge_array = np.ones(len(ne_psi_n))

    # if "omeg" in profiles.keys():
    #     omega_psi_n = profiles["omeg"]["psinorm"] * units.dimensionless
    #     omega_data = profiles["omeg"]["data"] * 1e3 * units.radians / units.second
    # else:
    #     omega_psi_n = te_psi_n * units.dimensionless
    #     omega_data = (
    #         np.zeros(len(omega_psi_n), dtype="float") * units.radians / units.second
    #     )

    # omega_func = UnitSpline(omega_psi_n, omega_data)

    # electron_charge = UnitSpline(
    #     ne_psi_n, -1 * unit_charge_array * units.elementary_charge
    # )

    # electron = Species(
    #     species_type="electron",
    #     charge=electron_charge,
    #     mass=electron_mass,
    #     dens=electron_dens_func,
    #     temp=electron_temp_func,
    #     omega0=omega_func,
    #     rho=rho_func,
    # )

    # result = {"electron": electron}
    # num_ions = len(species_name)

    # # Check whether fast particles.
    # try:
    #     if np.all(profiles["nb"]["data"] == 0.0):
    #         fast_particle = 0
    #     else:
    #         fast_particle = 1
    # except KeyError:
    #     fast_particle = 0

    # num_thermal_ions = num_ions - fast_particle

    # # thermal ions have same temperature in pFile.
    # ti_psi_n = profiles["ti"]["psinorm"] * units.dimensionless
    # ion_temp_data = profiles["ti"]["data"] * 1e3 * units.eV
    # ion_temp_func = UnitSpline(ti_psi_n, ion_temp_data)

    # for ion_it in np.arange(num_thermal_ions):
    #     if ion_it == num_thermal_ions - 1:
    #         ni_psi_n = profiles["ni"]["psinorm"] * units.dimensionless
    #         ion_dens_data = profiles["ni"]["data"] * 1e20 * units.meter**-3
    #         ion_dens_func = UnitSpline(ni_psi_n, ion_dens_data)

    #         ion_charge = species_name[ion_it]["Z"] * units.elementary_charge
    #         ion_nucleons = species_name[ion_it]["A"]
    #         ion_mass = ion_nucleons * deuterium_mass / 2.0

    #         species_name = ion_species_selector(ion_nucleons, ion_charge)

    #         result[species_name] = Species(
    #             species_type=species_name,
    #             charge=UnitSpline(ne_psi_n, ion_charge * unit_charge_array),
    #             mass=ion_mass,
    #             dens=ion_dens_func,
    #             temp=ion_temp_func,
    #             omega0=omega_func,
    #             rho=rho_func,
    #         )

    #     else:
    #         try:
    #             nz_psi_n = (
    #                 profiles[f"nz{ion_it+1}"]["psinorm"] * units.dimensionless
    #             )
    #             impurity_dens_data = (
    #                 profiles[f"nz{ion_it+1}"]["data"] * 1e20 * units.meter**-3
    #             )
    #         except KeyError:
    #             nz_psi_n = ni_psi_n
    #             impurity_dens_data = ion_dens_data * 0.0

    #         impurity_dens_func = UnitSpline(nz_psi_n, impurity_dens_data)

    #         impurity_charge = UnitSpline(
    #             ne_psi_n,
    #             species_name[ion_it]["Z"] * unit_charge_array * units.elementary_charge,
    #         )
    #         impurity_nucleons = species_name[ion_it]["A"]
    #         impurity_mass = impurity_nucleons * deuterium_mass / 2.0

    #         species_name = ion_species_selector(impurity_nucleons, impurity_charge)
    #         result[species_name] = Species(
    #             species_type=species_name,
    #             charge=impurity_charge,
    #             mass=impurity_mass,
    #             dens=impurity_dens_func,
    #             temp=ion_temp_func,
    #             omega0=omega_func,
    #             rho=rho_func,
    #         )

    # if fast_particle == 1:  # Adding the fast particle species_name.
    #     nb_psi_n = profiles["nb"]["psinorm"] * units.dimensionless
    #     fast_ion_dens_data = profiles["nb"]["data"] * 1e20 * units.meter**-3

    #     pb_psi_n = profiles["pb"]["psinorm"] * units.dimensionless
    #     fast_ion_press_data = profiles["pb"]["data"] * 1e3 * units.pascals

    #     if np.all(pb_psi_n != nb_psi_n):
    #         fast_ion_press_func = UnitSpline(pb_psi_n, fast_ion_press_data)
    #         fast_ion_press_data = fast_ion_press_func(nb_psi_n)

    #     fast_ion_temp_data = np_to_T(fast_ion_dens_data, fast_ion_press_data)

    #     fast_ion_dens_func = UnitSpline(nb_psi_n, fast_ion_dens_data)
    #     fast_ion_temp_func = UnitSpline(nb_psi_n, fast_ion_temp_data)

    #     fast_ion_charge = species_name[-1]["Z"] * units.elementary_charge
    #     fast_ion_nucleons = species_name[-1]["A"]
    #     fast_ion_mass = fast_ion_nucleons * deuterium_mass / 2.0

    #     fast_species = ion_species_selector(
    #         fast_ion_nucleons, fast_ion_charge
    #     ) + str("_fast")

    #     result[fast_species] = Species(
    #         species_type=fast_species,
    #         charge=UnitSpline(ne_psi_n, fast_ion_charge * unit_charge_array),
    #         mass=fast_ion_mass,
    #         dens=fast_ion_dens_func,
    #         temp=fast_ion_temp_func,
    #         omega0=omega_func,
    #         rho=rho_func,
    #     )

    # return Kinetics(kinetics_type="pFile", **result)

