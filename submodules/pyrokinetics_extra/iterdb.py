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
from pyrokinetics.kinetics import Kinetics
from pyrokinetics_extra.rho_profiles import species_dict_to_kinetics
from pyrokinetics.constants import deuterium_mass, electron_mass




class KineticsReaderITERDB(FileReader, file_type="ITERDB", reads=Kinetics):
    def read_from_file(self, file_path: PathLike, eq: Equilibrium = None) -> Kinetics:
        data = self.read_iterdb(file_path)
        iterdb_data = {'electrons':{'mass':electron_mass, 'charge':-1,'temp_profile':{'rho':data['TE                  eV']['RHOTOR              -'] , 'temp':data['TE                  eV']['TE                  eV']}, 'dens_profile':{'rho':data['NE                  m^-3']['RHOTOR              -'], 'dens':data['NE                  m^-3']['NE                  m^-3']}}, 'deuterium':{'mass':deuterium_mass, 'charge':+2,'temp_profile':{'rho':data['TE                  eV']['RHOTOR              -'] , 'temp':data['TE                  eV']['TE                  eV']}, 'dens_profile':{'rho':data['NM1                 m^-3']['RHOTOR              -'], 'dens':data['NM1                 m^-3']['NM1                 m^-3']}}}
        
        kinetics, psi_func = species_dict_to_kinetics(iterdb_data, eq)
        return kinetics

    def read_iterdb(self, file_path):
        data = {}
        with open(file_path, 'r') as file:
            line = file.readline()
            while line:
                headder = ''
                # read the first headder
                while line:
                    if ';' in line or '***' in line:
                        headder = headder + line
                    else: 
                        break
                    line = file.readline()
                # print('HEADDER', headder)
                if not line: break

                # Search for the pattern in the text
                x_label = re.search(r'^(.*?);-INDEPENDENT VARIABLE LABEL: X-', headder, re.MULTILINE).group(1).strip()
                num_x = int(re.search(r'^(.*?);-# OF X PTS-', headder, re.MULTILINE).group(1).strip())
                y_label = re.search(r'^(.*?);-INDEPENDENT VARIABLE LABEL: Y-', headder, re.MULTILINE).group(1).strip()
                num_y = int(re.search(r'^(.*?);-# OF Y PTS-', headder, re.MULTILINE).group(1).strip())
                z_label = re.search(r'^(.*?);-DEPENDENT VARIABLE LABEL-', headder, re.MULTILINE).group(1).strip()
                num_z = num_x * num_y

                species_profiles = ''
                while line:
                    if ';' not in line:
                        species_profiles += line
                    else:
                        break
                    line = file.readline()
                # print('species profile', species_profiles)
                species_profiles = np.array(re.findall(r'-?\d+\.\d+E[+-]?\d+', species_profiles)).astype('float')
                data[z_label] = {x_label:species_profiles[0:num_x], y_label:species_profiles[num_x:num_x+num_y], z_label:species_profiles[num_x+num_y:]}

        return data