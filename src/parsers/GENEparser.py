import os
import math
import f90nml
import numpy as np
import json
from .base import Parser
import fortranformat as ff
import scipy
from pylab import *
from sys import argv,exit,stdout
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as US
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import CubicSpline

import f90nml
import re
from collections import deque


#import from submodules
try: 
    from IFS_scripts.geomWrapper import calc_kperp_omd, init_read_geometry_file
    from IFS_scripts.parIOWrapper import init_read_parameters_file

    from TPED.projects.GENE_sim_reader.archive.ARCHIVE_GENE_field_data import GeneField as GF
    from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder as GFF
except: ModuleNotFoundError("You need to add sys.path.append(relative/dir/submodules) to your script or notebook for these to be imported.\n For example if in the notebooks dir sys.path.append('../submodules')")


from parsers.HELENAparser import *
HELENAparser = HELENAparser()
# import subprocess

mu_0 = 4e-7 * np.pi
ElectronCharge = 1.6022e-19

class GENEparser(Parser):
    """_summary_

    Args:
        Parser (_type_): _description_
    """
    
    def __init__(self):
        """
        Initializes the GENEparser object.

        """
        #self.default_namelist = ""
        pass
    
    def write_input_file(self, params: dict, run_dir: str):
        """Pass"""
        
        pass  

    def write_parameters_file(self, original_file_path, new_file_path, updates):
        """
        Updates specific parameters in the parameter file.
        
        :param original_file_path: Path to the original file.
        :param new_file_path: Path to the new file with updated parameters.
        :param updates: Dictionary of updates, e.g., {'in_out': {'diagdir': '/new/path'}}
        """
        with open(original_file_path, 'r') as file:
            lines = file.readlines()
        
        with open(new_file_path, 'w') as new_file:
            current_section = None
            for line in lines:
                if line.strip().startswith('&'):
                    current_section = line.strip()[1:].strip()
                    new_file.write(line)
                elif line.strip().startswith('/'):
                    current_section = None
                    new_file.write(line)
                elif current_section and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    if current_section in updates and key in updates[current_section]:
                        new_file.write(f"{key} = {updates[current_section][key]}\n")
                    else:
                        new_file.write(line)
                else:
                    new_file.write(line)

    def read_parameters_file(self, file_path):
        """
        Reads a parameter file and stores parameters in a dictionary by section.
        """
        parameters = {}
        current_section = None
        
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('&'):
                    current_section = line[1:].strip()
                    parameters[current_section] = {}
                elif line.startswith('/'):
                    current_section = None
                elif current_section and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    parameters[current_section][key] = value
                elif current_section and line.startswith('!'):
                    # Ignore comment lines inside sections
                    continue
        
        return parameters
        
    def read_EFIT_file(self, efit_file_name: str):

        f = open(efit_file_name,'r')
        eqdsk=f.readlines()
        line1=eqdsk[0].split()
        nw=int(eqdsk[0].split()[-2])
        nh=int(eqdsk[0].split()[-1])
        print ('EFIT file Resolution: %d x %d' %(nw,nh))

        entrylength=16
        #note: here rmin is rleft from EFIT
        try:
            rdim,zdim,rctr,rmin,zmid=[float(eqdsk[1][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[1])/entrylength))]
        except:
            entrylength=15
            try:
                rdim,zdim,rctr,rmin,zmid=[float(eqdsk[1][j*entrylength:(j+1)*entrylength]) for j in range(len(eqdsk[1])/entrylength)]
            except:
                exit('Error reading EQDSK file, please check format!')

        rmag,zmag,psiax,psisep,Bctr=[float(eqdsk[2][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[2])/entrylength))]
        dum,psiax2,dum,rmag2,dum=[float(eqdsk[3][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[3])/entrylength))]
        zmag2,dum,psisep2,dum,dum=[float(eqdsk[4][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[4])/entrylength))]
        if rmag!=rmag2: sys.exit('Inconsistent rmag: %7.4g, %7.4g' %(rmag,rmag2))
        if psiax2!=psiax: sys.exit('Inconsistent psiax: %7.4g, %7.4g' %(psiax,psiax2))
        if zmag!=zmag2: sys.exit('Inconsistent zmag: %7.4g, %7.4g' %(zmag,zmag2) )
        if psisep2!=psisep: sys.exit('Inconsistent psisep: %7.4g, %7.4g' %(psisep,psisep2))

        ###print "rmag", rmag
        ###print "zmag", zmag
        ###print "psiax", psiax
        ###print "psisep", psisep
        ###print "Bctr", Bctr
        # (R,Z) grid on which psi_pol is written
        Rgrid = np.arange(nw)/float(nw-1)*rdim+rmin
        ###print "rdim",rdim
        ###print "rmin",rmin
        ###print "first few Rgrid points", Rgrid[0:6]
        ###print "last few Rgrid points", Rgrid[-7:-1]
        Zgrid = np.arange(nh)/float(nh-1)*zdim+(zmid-zdim/2.0)
        ###print "zdim",zdim
        ###print "zmid",zmid
        ###print "first few Zgrid points", Zgrid[0:6]
        ###print "last few Zgrid points", Zgrid[-7:-1]

        # F, p, ffprime, pprime, q are written on uniform psi_pol grid
        # uniform grid of psi_pol~[psiax,psisep], resolution=nw
        F=empty(nw,dtype=float)
        p=empty(nw,dtype=float)
        ffprime=empty(nw,dtype=float)
        pprime=empty(nw,dtype=float)
        qpsi=empty(nw,dtype=float)
        # psi_pol is written on uniform (R,Z) grid (res=nw(R)*nh(Z))
        psirz_1d=empty(nw*nh,dtype=float)
        
        start_line=5
        lines=range(int(nw/5))
        if nw%5!=0: lines=range(int(nw/5)+1)
        for i in lines:
            n_entries=int(len(eqdsk[i+start_line])/entrylength)
            F[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
        start_line=i+start_line+1

        for i in lines:
            n_entries=int(len(eqdsk[i+start_line])/entrylength)
            p[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
        start_line=i+start_line+1

        for i in lines:
            n_entries=int(len(eqdsk[i+start_line])/entrylength)
            ffprime[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
        start_line=i+start_line+1

        for i in lines:
            n_entries=int(len(eqdsk[i+start_line])/entrylength)
            pprime[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
        start_line=i+start_line+1

        lines_twod=range(int(nw*nh/5))
        if nw*nh%5!=0: lines_twod=range(int(nw*nh/5)+1)
        for i in lines_twod:
            n_entries=int(len(eqdsk[i+start_line])/entrylength)
            psirz_1d[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
        start_line=i+start_line+1
        psirz=psirz_1d.reshape(nh,nw)

        for i in lines:
            n_entries=int(len(eqdsk[i+start_line])/entrylength)
            qpsi[i*5:i*5+n_entries]=[float(eqdsk[i+start_line][j*entrylength:(j+1)*entrylength]) for j in range(n_entries)]
        start_line=i+start_line+1

        # even grid of psi_pol, on which all 1D fields are defined
        psip_n = np.linspace(0.0,1.0,nw)
        # return data read from efit file 
        # psip_n: uniform flux grid from magnetic axis to separatrix
        # F, p, ffprime, pprime, qpsi are on psip_n
        # uniform (R,Z) grid, psirz is on this grid
        return psip_n, Rgrid, Zgrid, F, p, ffprime, pprime, psirz, qpsi, rmag, zmag, nw,psiax,psisep

    def calc_rho_tor(self, psip_n, psiax, psisep, qpsi, nw,psip_n_max=0.999):

        #create rho_tor grid on even psi_pol grid
        interpol_order = 3 
        psi_pol = np.empty(len(psip_n))
        for i in range(len(psip_n)):
            psi_pol[i] = psiax+psip_n[i]*(psisep-psiax)

        q_spl_psi = US(psi_pol, qpsi, k=interpol_order, s=1e-5)
        psi_pol_fine = linspace(psi_pol[0], psi_pol[-1], nw*10)
        psi_tor_fine = empty((nw*10),dtype=float)
        psi_tor_fine[0] = 0.
        qfine = np.empty(nw*10)
        ##################
        ##################
        qnew = q_spl_psi(psi_pol_fine)
        psip_n_fine = (psi_pol_fine-psiax)/(psisep-psiax)
        #plt.plot(psip_n_fine,qnew)
        #plt.plot(psip_n,qpsi,'x-')
        #plt.show()
        ##################
        ##################
        for i in range(1,nw*10):
            x=psi_pol_fine[:i+1]
            y=q_spl_psi(x)
            psi_tor_fine[i]=np.trapz(y,x)

        phi_edge = psi_tor_fine[-1]
        rhot_n_fine=np.sqrt(psi_tor_fine/(psi_tor_fine[-1]-psi_tor_fine[0]))
        rho_tor_spl=US(psi_pol_fine, rhot_n_fine, k=interpol_order, s=1e-5)
        # rhot_n grid (not uniform, on even grid of psi_pol) of resolution=nw 
        rhot_n=rho_tor_spl(psi_pol)
        
        # rho_tor_spl takes value of psi_pol (not normalized) and convert into rhot_n
        return rho_tor_spl, rhot_n, phi_edge

    def calc_B_fields(self, Rgrid, rmag, Zgrid, zmag, psirz, psiax, psisep, F, nw, psip_n):
        # Z0_ind is the index of Zgrid of midplane
        Z0_ind = np.argmin(np.abs(Zgrid-zmag))
        # psi_midplane is psi_pol at midplane on even Rgrid
        psi_pol_mp = psirz[Z0_ind,:]
        # Rmag_ind is index of unif_R at rmag
        Rmag_ind = np.argmin(np.abs(Rgrid - rmag))
        ###print "rmag",rmag
        ###print "Rmag_ind",Rmag_ind
        ###print "Rgrid[Rmag_ind]~rmag", Rgrid[Rmag_ind]
        ###print "psi_pol_mp[Rmag_ind]~psiax", psi_pol_mp[Rmag_ind]
        psi_pol_obmp = psi_pol_mp[Rmag_ind:].copy()
        #normalize psi_pol_obmp to psip_n_temp
        psip_n_temp = np.empty(len(psi_pol_obmp))
        for i in range(len(psi_pol_obmp)):
            psip_n_temp[i] = (psi_pol_obmp[i]-psiax)/(psisep-psiax)
        unif_R = np.linspace(Rgrid[Rmag_ind],Rgrid[-1],nw*10)
    #    unif_R = np.linspace(Rgrid[Rmag_ind],Rgrid[-1],nw)
        psip_n_unifR = self.interp(Rgrid[Rmag_ind:],psip_n_temp,unif_R)
        psisep_ind = np.argmin(abs(psip_n_unifR-1.02))
        ###print "psisep_ind", psisep_ind
        ###print "psip_n_temp[psisep_ind]~1", psip_n_unifR[psisep_ind]
        #print "we have a problem here because uniform R grid doesn't have enough resolution near separatrix"
        psip_n_obmp = psip_n_unifR[:psisep_ind].copy()
        ###print "psip_n_obmp[0]~0", psip_n_obmp[0]
        ###print "psip_n_obmp[-1]~1", psip_n_obmp[-1]
        #plt.plot(psi_pol_obmp)
        #plt.show()
        R_obmp = unif_R[:psisep_ind].copy()
        # B_pol is d psi_pol/ d R * (1/R)
        #B_pol = fd_d1_o4(psi_pol_obmp, unif_R[Rmag_ind:Rmag_ind+psisep_ind])/unif_R[Rmag_ind:Rmag_ind+psisep_ind]
        B_pol = self.fd_d1_o4(psip_n_obmp*(psisep-psiax)+psiax,R_obmp)/R_obmp
        # convert F(on even psi_pol grid) to F(on even R grid)
        F_obmp = self.interp(psip_n, F, psip_n_obmp)
        # B_tor = F/R
        B_tor = F_obmp/R_obmp
        
        # next part added to get past interpolation issue
        
        differences = np.diff(psip_n_obmp)
    
        # Find indices where the difference is positive (strictly increasing)
        increasing_indices = np.where(differences > 0)[0]
        
        # Since np.diff reduces the length of the array by 1, we need to adjust indices
        increasing_indices = increasing_indices + 1
        
        psip_n_obmp = psip_n_obmp[increasing_indices]
        R_obmp = R_obmp[increasing_indices]
        B_pol = B_pol[increasing_indices]
        B_tor = B_tor[increasing_indices]
        # psip_n_obmp is normalized psi_pol at outboard midplane on uniform unif_R
        # B_tor and B_pol are on uniform unif_R as well
        # psip_n_obmp is unlike psip_n ([0,1]), it goes from 0 to 1.06 here
        return psip_n_obmp, R_obmp, B_pol, B_tor

    def read_EFIT_parameters(self, efit_file_name):

        f = open(efit_file_name,'r')
        eqdsk=f.readlines()
        line1=eqdsk[0].split()
        nw=int(eqdsk[0].split()[-2])
        nh=int(eqdsk[0].split()[-1])

        entrylength=16
        #note: here rmin is rleft from EFIT
        try:
            rdim,zdim,rctr,rmin,zmid=[float(eqdsk[1][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[1])/entrylength))]
        except:
            entrylength=15
            try:
                rdim,zdim,rctr,rmin,zmid=[float(eqdsk[1][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[1])/entrylength))]
            except:
                exit('Error reading EQDSK file, please check format!')

        rmag,zmag,psiax,psisep,Bctr=[float(eqdsk[2][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[2])/entrylength))]
        curr,psiax2,dum,rmag2,dum=[float(eqdsk[3][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[3])/entrylength))]
        zmag2,dum,psisep2,dum,dum=[float(eqdsk[4][j*entrylength:(j+1)*entrylength]) for j in range(int(len(eqdsk[4])/entrylength))] 
        if rmag!=rmag2: sys.exit('Inconsistent rmag: %7.4g, %7.4g' %(rmag,rmag2))
        if psiax2!=psiax: sys.exit('Inconsistent psiax: %7.4g, %7.4g' %(psiax,psiax2))
        if zmag!=zmag2: sys.exit('Inconsistent zmag: %7.4g, %7.4g' %(zmag,zmag2) )
        if psisep2!=psisep: sys.exit('Inconsistent psisep: %7.4g, %7.4g' %(psisep,psisep2))

        return rdim,zdim,rctr,rmin,zmid,Bctr,curr,nh

    def get_dimpar_pars(self, efit_file_name,rhot0):

        psip_n, Rgrid, Zgrid, F, p, ffprime, pprime, psirz, qpsi, rmag, zmag, nw,psiax,psisep = self.read_EFIT_file(efit_file_name)
        #plt.plot(Rgrid)
        #plt.title('Rgrid (rmag = '+str(rmag)+')')
        #plt.show()
        R_major = rmag
        dummy, rhot_n, phi_edge = self.calc_rho_tor(psip_n, psiax, psisep, qpsi, nw,psip_n_max=0.999)
        psip_n_obmp, R_obmp, B_pol, B_tor = self.calc_B_fields(Rgrid, rmag, Zgrid, zmag, psirz, psiax, psisep, F, nw, psip_n)
        Bref = abs(B_tor[0])
        Lref = np.sqrt(2.0*abs(phi_edge/Bref))
        #plt.plot(rhot_n,psip_n)
        #plt.xlabel('rhot_n')
        #plt.ylabel('psi_n')
        #plt.show()
        irhot_n = np.argmin(abs(rhot_n-rhot0))
        q0 = qpsi[irhot_n]
        return Lref, Bref, R_major, q0

    def get_current_density(self, efit_file_name):

        psip_n, Rgrid, Zgrid, F, p, ffprime, pprime, psirz, qpsi, rmag, zmag, nw,psiax,psisep = self.read_EFIT_file(efit_file_name)
        Jtot = Rgrid*pprime+ffprime/Rgrid
        return psip_n,Rgrid,Jtot


    def get_geom_pars(self, efit_file_name,rhot0):
        psip_n, Rgrid, Zgrid, F, p, ffprime, pprime, psirz, qpsi, rmag, zmag, nw,psiax,psisep = self.read_EFIT_file(efit_file_name)
        R_major = rmag
        dummy, rhot_n, phi_edge = self.calc_rho_tor(psip_n, psiax, psisep, qpsi, nw,psip_n_max=0.999)
        psip_n_obmp, R_obmp, B_pol, B_tor = self.calc_B_fields(Rgrid, rmag, Zgrid, zmag, psirz, psiax, psisep, F, nw, psip_n)
        Bref = abs(B_tor[0])
        Lref = np.sqrt(2.0*abs(phi_edge/Bref))
        irhot_n = np.argmin(abs(rhot_n-rhot0))
        q0 = qpsi[irhot_n]


        rhot_new = np.linspace(rhot_n[0],rhot_n[-1],4*len(rhot_n))
        qpsi_new = self.interp(rhot_n,qpsi,rhot_new)  
        shat = rhot_new/qpsi_new*self.fd_d1_o4(qpsi_new,rhot_new)
        irhot_new = np.argmin(abs(rhot_new-rhot0))
        shat0 = shat[irhot_new]
        return Lref, Bref, R_major, q0, shat0
    
    def fd_d1_o4(self, var,grid,mat=False):
        """Centered finite difference, first derivative, 4th order.
        var: quantity to be differentiated.
        grid: grid for var 
        mat: matrix for the finite-differencing operator. if mat=False then it is created"""

        if not mat:
            mat=self.get_mat_fd_d1_o4(len(var),grid[1]-grid[0])

        dvar=-np.dot(mat,var)
        dvar[0]=0.0
        dvar[1]=0.0
        #dvar[2]=0.0
        dvar[-1]=0.0
        dvar[-2]=0.0
        #dvar[-3]=0.0
        return -dvar 

    def get_mat_fd_d1_o4(self, size,dx,plot_matrix=False):
        """Creates matrix for centered finite difference, first derivative, 4th order.
        size: size of (number of elements in) quantity to be differentiated
        dx: grid spacing (for constant grid)."""

        prefactor=1.0/(12.0*dx)
        mat=np.zeros((size,size),dtype='float')    
        for i in range(size):
            if i-1 >= 0:
                mat[i,i-1]=-8
            if i-2 >= 0:
                mat[i,i-2]=1
            if i+1 <= size-1:
                mat[i,i+1]=8
            if i+2 <= size-1:
                mat[i,i+2]=-1
    
        mat=prefactor*mat

        if plot_matrix:
            plt.contourf(mat,50)
            plt.colorbar()
            plt.show()

        return mat
    
    def interp(self, xin,yin,xnew):
        """
        xin: x variable input
        yin: y variable input
        xnew: new x grid on which to interpolate
        yout: new y interpolated on xnew
        """

        #splrep returns a knots and coefficients for cubic spline
        rho_tck = interpolate.splrep(xin,yin)
        #Use these knots and coefficients to get new y
        yout = interpolate.splev(xnew,rho_tck,der=0)

        return yout
    
    def calc_er_vrot(self, profile_data, efit_file_name):
        #data = np.genfromtxt(gene_profiles_file_name)

        rhot = profile_data[:,0]
        psi = profile_data[:,1]**2
        Ti = profile_data[:,2]/10**3  # [keV]
        ni = profile_data[:,3]/10**19 # [10^19 m^-3]

        mi = 1.673e-27  # [kg]
        ee = 1.602e-19  # [C]
        mref = 2.0 * mi
        Z = 1.0

        # Read dimensional and magnetic parameters from the EFIT file
        Lref, Bref, R_major, q0 = self.get_dimpar_pars(efit_file_name, 0.9)
        psip_n, Rgrid, Zgrid, F, p, ffprime, pprime, psirz, qpsi, rmag, zmag, nw, psiax, psisep = self.read_EFIT_file(efit_file_name)

        psisep0 = psisep-psiax
        
        psi0 = np.linspace(0.0, 1.0, num=3000)
        ni0 = self.interp(psi, ni, psi0)
        Ti0 = self.interp(psi, Ti, psi0)
        Ti0J = Ti0 * 1000.0 * ee  # Convert to Joules
        ni00 = ni0 * 10**19       # Convert to m^-3
        pi0 = Ti0J * ni00
        rhot0 = self.interp(psi, rhot, psi0)
        qpsi0 = self.interp(psip_n, qpsi, psi0)

        R0 = self.interp(psip_n, Rgrid, psi0)
        trpeps = rhot0 * Lref / R_major
        coll = 2.3031E-5 * Lref * ni0 / (Ti0**2) * (24.0 - np.log(np.sqrt(ni0 * 1.0E13) / Ti0 * 0.001))
        nustar_i = (8.0 / 3.0) * (1 / np.pi**0.5) * qpsi0 / trpeps**1.5 * (R_major / Lref) * Z**4 * coll

        ft = trpeps**0.5 * (1.0 + 0.46 * (1.0 - trpeps))
        fc = 1.0 - ft
        a = 1.0 / (1.0 + 0.5 * nustar_i**0.5)
        b = -1.17 * fc / (1.0 - 0.22 * ft - 0.19 * ft**2) + 0.25 * (1 - ft**2) * nustar_i**0.5
        c = 0.315 * nustar_i**2 * ft**6
        d = 1.0 / (1.0 + 0.15 * nustar_i**2 * ft**6)
        kpar = -(a * b + c) * d

        dTdpsi = self.fd_d1_o4(Ti0J, psi0)
        dndpsi = self.fd_d1_o4(ni00, psi0)
        dpdpsi = self.fd_d1_o4(pi0, psi0)

        omegator = (1.0 / psisep0 / ee) * ((1 - kpar) * dTdpsi + Ti0J / ni00 * dndpsi)

        # Calculate magnetic fields
        psip_n_obmp, R_obmp, B_pol, B_tor = self.calc_B_fields(Rgrid, rmag, Zgrid, zmag, psirz, psiax, psisep, F, nw, psip_n)
        B_pol0 = self.interp(psip_n_obmp, B_pol, psi0)
        B_tor0 = self.interp(psip_n_obmp, B_tor, psi0)

        Er = omegator * B_pol0 * R_major

        B2 = B_pol0**2 + B_tor0**2
        
        vpol = -Er * B_tor0 / B2  # [m/s]
        vtor = Er * B_pol0 / B2   # [m/s]
        #two different estimates for vrot since only toroidal rotation can be included
        #upper: all flows in vtor direction
        #lower: only vtor flow component considered
        vrot_upper = (vpol + vtor) / R_major # [rad/s]
        vrot_lower = vtor / R_major          # [rad/s]
        """""
        plt.figure(figsize=(5,4))
        plt.plot(rhot0, vtor/1000, color="red", label = "MAST-U_49108")
        plt.xlim(0.85, 1)
        plt.xlabel(r'$\rho_{tor}$')
        plt.ylabel(r'$v_{ExB_{tor}\times 10^3} \, [\frac{m}{s}]$') 
        plt.legend()
        plt.show()
        """""
        return rhot0, omegator, Er, vrot_upper, vrot_lower

    def create_T_n_rho_grid(self, helena_output_dir: str):
        eliteinp_path_name = os.path.join(helena_output_dir, "eliteinp")
        elite_data = HELENAparser.read_eliteinput(eliteinp_path_name)
        psi = elite_data["Psi"]
        q = elite_data["q"]
        Ti = elite_data["Ti"]
        ne = elite_data["ne"]
        nMainIon = elite_data["nMainIon"]
        
        dpsi = np.diff(psi)
        q_cd = 0.5 * (q[:-1] + q[1:])
        dpsi = np.diff(psi)
        dphit = q_cd * dpsi
        phit = np.cumsum(dphit)
        phitt = np.concatenate(([0], phit))
        rhot = np.sqrt(phitt / np.max(phitt))
        rhop = np.sqrt(psi /np.max(psi))
        
        data = np.column_stack((rhot[1:-1], rhop[1:-1], Ti[1:-1], nMainIon[1:-1]))
        return data
    
    def write_iterdb_profiles(
        self, helena_output_dir: str, eqdsk_dir:str, iterdb_output_path: str
    ) -> None:
        fpath_elite = os.path.join(helena_output_dir, "eliteinp")
        elite_data = HELENAparser.read_eliteinput(fpath_elite)
        #path_vrot = os.path.join(helena_output_dir, "profiles_vrot.dat")
        f_out = open(iterdb_output_path, "w")
        format_iterdb = ff.FortranRecordWriter("(6e13.6)")

        def write_header(label, unit, nx, ny=1):
            f_out.write(
                "  00000DUM  "
                + str(ny + 1)
                + " 0 6              ;-SHOT #- F(X,Y) DATA -UFILELIB- 00Jan0000\n"
            )
            f_out.write(
                "                               ;-SHOT DATE-  UFILES ASCII FILE SYSTEM\n"
            )
            f_out.write(
                "   0                           ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES-\n"
            )
            f_out.write(
                " RHOTOR              -         ;-INDEPENDENT VARIABLE LABEL: X-\n"
            )
            f_out.write(
                " TIME                SECONDS   ;-INDEPENDENT VARIABLE LABEL: Y-\n"
            )
            f_out.write(
                " "
                + str(label)
                + "               "
                + str(unit)
                + "     ;-DEPENDENT VARIABLE LABEL-\n"
            )
            f_out.write(
                " 0                             ;-PROC CODE- 0:RAW 1:AVG 2:SM. 3:AVG+SM\n"
            )
            f_out.write("        " + str(nx) + "                    ;-# OF X PTS-\n")
            f_out.write(
                "          "
                + str(ny)
                + "                    ;-# OF Y PTS-   X,Y,F(X,Y) DATA FOLLOW:\n"
            )

        def write_closure():
            f_out.write(";----END-OF-DATA-----------------COMMENTS:----------\n")
            f_out.write(
                "*******************************************************************************\n"
            )
            f_out.write(
                "*******************************************************************************\n"
            )

        def write_quantity(vector):
            number = math.ceil(len(vector) / 6)
            for i in range(number):
                f_out.write(" " + format_iterdb.write(vector[6 * i : 6 * i + 6]) + "\n")
            f_out.write(" " + format_iterdb.write([0.0]) + "\n")

        psi = elite_data["Psi"]
        nx = len(psi)
        te = elite_data["Te"]
        ti = elite_data["Ti"]
        ne = elite_data["ne"]
        q = elite_data["q"]
        q_cd = 0.5 * (q[:-1] + q[1:])
        dpsi = np.diff(psi)
        dphit = q_cd * dpsi
        phit = np.cumsum(dphit)
        phitt = np.concatenate(([0], phit))
        rhot = np.sqrt(phitt / np.max(phitt))
        
        profile_grid = self.create_T_n_rho_grid(helena_output_dir)
        
        rhot0, omegator, Er, all_vrot_upper, vrot_lower = self.calc_er_vrot(profile_grid, eqdsk_dir)
        filtered_indices = vrot_lower != 0
        rhot0 = rhot0[filtered_indices]
        vrot_lower = vrot_lower[filtered_indices]        
        interpolation_function = scipy.interpolate.interp1d(rhot0, vrot_lower, fill_value="extrapolate")
        
        vrot_lower = interpolation_function(rhot)
        
        if rhot[0] == 0:
            rhot = rhot[1:]
            te = te[1:]
            ti = ti[1:]
            ne = ne[1:]
            vrot_lower = vrot_lower[1:]
        
        if rhot[-1] == 1:
            rhot = rhot[:-1]
            te = te[:-1]
            ti = ti[:-1]
            ne = ne[:-1]
            vrot_lower = vrot_lower[1:]       
        
        nx = len(rhot)
        write_header("TE   ", "eV   ", nx)
        write_quantity(rhot)
        write_quantity(te)
        write_closure()
        write_header("TI   ", "eV   ", nx)
        write_quantity(rhot)
        write_quantity(ti)
        write_closure()
        write_header("NE   ", "m^-3 ", nx)
        write_quantity(rhot)
        write_quantity(ne)
        write_closure()
        write_header("NM1  ", "m^-3 ", nx)
        write_quantity(rhot)
        write_quantity(ne)
        write_closure()
        write_header("VROT ", "rad/s", nx)
        write_quantity(rhot)
        write_quantity(vrot_lower)
        write_closure()
        write_header("ZEFFR", "     ", nx)
        write_quantity(rhot)
        write_quantity(np.ones(len(ne)))
        write_closure()
        f_out.close()
    
    def helena_to_eqdsk(
        self, helena_output_dir: str, eqdsk_output_path: str, NR: int, NZ: int
    ) -> None:
        fpath_elite = os.path.join(helena_output_dir, "eliteinp")
        fpath_f12 = os.path.join(helena_output_dir, "fort.12")
        fpath_f20 = os.path.join(helena_output_dir, "fort.20")
        elite_data = HELENAparser.read_eliteinput(fpath_elite)
        geometry_vars = HELENAparser.get_real_world_geometry_factors_from_f20(fpath_f20)
        helena_fort12_outputs = HELENAparser.read_helena_fort12(
            fpath_f12,
            B0=geometry_vars["BVAC"],
            RVAC=geometry_vars["RVAC"],
            CPSURF=geometry_vars["CPSURF"],
            RADIUS=geometry_vars["RADIUS"],
        )
        psi_arr_og = elite_data["Psi"]#[1:-1]
        psi_arr_og = psi_arr_og - np.max(psi_arr_og)
        max_value = np.max(psi_arr_og)
        fpol_arr_og = elite_data["fpol"]#[1:-1]#[1:]
        pres_arr_og = helena_fort12_outputs["P0_SCALED"]#[1:-1]
        ffprim_arr_og = elite_data["ffp"]#[1:-1]
        qpsi_arr_og = elite_data["q"]#[1:-1]
        r_arr_og = elite_data["R"]#[:, 1:-1]
        z_arr_og = elite_data["z"]#[:, 1:-1]
        RDIM = (r_arr_og.max() - r_arr_og.min())*1.1
        ZDIM = (z_arr_og.max() - z_arr_og.min())*1.1
        ZMID = z_arr_og[0, 0]#z_arr_og.max() - ZDIM/2 #-ZDIM / 2.0 #was 0.0
        RLEFT = (r_arr_og.min())-(r_arr_og.max() - r_arr_og.min())*0.05
 

        SIMAG = psi_arr_og.min() #(psi_arr_og.max())#-psi_arr_og.min()) #psi_arr_og.min()
        SIBRY = psi_arr_og.max() #0
        RMAXIS = geometry_vars["RMAGAXIS"]
        RCENTR = geometry_vars["RVAC"]
        ZMAXIS = ZMID  # NOTE: Assumption
        BCENTR = geometry_vars["BVAC"]
        CURRENT = geometry_vars["CURRENT"]
        
        RBBBS = r_arr_og[:, -1] #elite_data["R"][:, -1]
        ZBBBS = z_arr_og[:, -1] #elite_data["z"][:, -1]
        NBBBS = RBBBS.shape[0] 

        limitr = NBBBS
        RLIM = np.zeros_like(RBBBS)
        ZLIM = np.zeros_like(ZBBBS)

        """ Remap to uniform psin grid """
        psi_arr = np.linspace(SIMAG, SIBRY, NR)
        #print("pri_arr", psi_arr)
        def remap_func_to_new_domain(
            x_old, y_old, x_new
        ) -> tuple[np.ndarray, np.ndarray]:
            spl_f = scipy.interpolate.CubicSpline(x_old, y_old)
            return spl_f(x_new), spl_f(x_new, 1)

        fpol_new, _ = remap_func_to_new_domain(psi_arr_og, fpol_arr_og, psi_arr)
        pres_new, pprime_new = remap_func_to_new_domain(
            psi_arr_og, pres_arr_og, psi_arr
        )
        ffpr_new, _ = remap_func_to_new_domain(psi_arr_og, ffprim_arr_og, psi_arr)
        qpsi_new, _ = remap_func_to_new_domain(psi_arr_og, qpsi_arr_og, psi_arr)

        """ Remap psi to R, Z on uniform R, Z grid"""

        r_1d = np.linspace(RLEFT, RLEFT + RDIM, NR)
        z_1d = np.linspace(ZMID - ZDIM / 2.0, ZMID + ZDIM / 2.0, NZ)
        
        R_new, Z_new = np.meshgrid(r_1d, z_1d)
        psi_old_2d = np.repeat(psi_arr_og[:, np.newaxis], z_arr_og.shape[0], axis=1).T      
        
        fill_value = (
            psi_old_2d.max() + 0.2
        )  # NOTE The plus value is to ensure that the boundary contour is found,
        # or that there is a "boundary" value.
        fill_value = 1   
        grid_lin = scipy.interpolate.griddata(
            np.array([r_arr_og.flatten(), z_arr_og.flatten()]).T,
            psi_old_2d.flatten().T,
            (R_new, Z_new),
            method="linear",
            fill_value=fill_value,
        )
        grid_lin = grid_lin.T
        grid_lin_in = grid_lin[np.where(grid_lin<fill_value)]
        max_grid_lin = np.max(grid_lin_in)
        R_boundary = []
        Z_boundary = []
        grid_lin_boundary = []
        
        # Check for boundary points; iterate through the grid
        for i in range(NZ):
            for j in range(NR):
                if grid_lin[j, i] != 1:
                    # Check the surrounding points for a transition
                    if (grid_lin[j-1, i] == fill_value or grid_lin[j+1, i] == fill_value or
                        grid_lin[j, i-1] == fill_value or grid_lin[j, i+1] == fill_value or
                        grid_lin[j-1, i-1] == fill_value or grid_lin[j-1, i+1] == fill_value or
                        grid_lin[j+1, i-1] == fill_value or grid_lin[j+1, i+1] == fill_value):
                        R_boundary.append(R_new[i, j])
                        Z_boundary.append(Z_new[i, j])
                        grid_lin_boundary.append(grid_lin[j, i])
        
        R_boundary = np.array(R_boundary)
        Z_boundary = np.array(Z_boundary)

        # Process each grid point that has the fill_value        
        RBBBS2 = r_arr_og[:,-100]
        ZBBBS2 = z_arr_og[:,-100]
        PSIBS2 = psi_old_2d[:,-100]
        PSIBS = psi_old_2d[:,-1]
        delta_R = RBBBS - RBBBS2
        delta_Z = ZBBBS - ZBBBS2
        d_21 = np.sqrt(delta_R**2 + delta_Z**2)
        PSI21 = PSIBS - PSIBS2
        RBBBS2_2, _ = remap_func_to_new_domain(range(len(RBBBS2)), RBBBS2, np.linspace(0, len(RBBBS), 10000))
        ZBBBS2_2, _ = remap_func_to_new_domain(range(len(ZBBBS2)), ZBBBS2, np.linspace(0, len(ZBBBS), 10000))
        RBBBS_2, _ = remap_func_to_new_domain(range(len(RBBBS2)), RBBBS, np.linspace(0, len(RBBBS), 10000))
        ZBBBS_2, _ = remap_func_to_new_domain(range(len(ZBBBS2)), ZBBBS, np.linspace(0, len(ZBBBS), 10000)) 
        d_21_2 = np.zeros(len(RBBBS_2))

        for i in range(len(RBBBS_2)):
            # Compute distances from point i on the first surface to all points on the second surface
            distances = np.sqrt((RBBBS_2[i] - RBBBS2_2)**2 + (ZBBBS_2[i] - ZBBBS2_2)**2)
            # Find the minimum distance and store it in d_21_2
            d_21_2[i] = np.min(distances)

        # d_21_2 now contains the distance from each point on the first surface to the closest point on the second surface
        for i in range(grid_lin.shape[1]):
            for j in range(grid_lin.shape[0]):
                if grid_lin[j, i] == fill_value:
                    dists = np.array(np.sqrt(((R_boundary) - R_new[i, j])**2 + ((Z_boundary) - Z_new[i, j])**2))
                    d = min(dists)
                    clos_i = np.where(dists == d)
                    
                    dists2 = np.array(np.sqrt((RBBBS_2 - R_new[i, j])**2 + (ZBBBS_2 - Z_new[i, j])**2))
                    d2 = min(dists2)
                    clos_i2 = np.where(dists2 == d2)[0]

                    grid_lin[j, i] = grid_lin_boundary[int(clos_i[0])] + PSI21[1]/d_21_2[int(clos_i2[0])]*d

        
        #SIMAG = -(psi_arr_og.max())#-psi_arr_og.min())
        #SIBRY = 0
        #grid_lin = grid_lin - max_grid_lin #+ SIMAG #max_grid_lin
        
        format_2000 = ff.FortranRecordWriter("(6a8, 3i4)")
        format_2020 = ff.FortranRecordWriter("(5e16.9)")
        format_2022 = ff.FortranRecordWriter("2i5")

        XDUM = 0 # was -1.0
        #SIBRY = 0
        idnum = 3
        case_strings = ["HELENA", "PRODUCED", "EQDSK", "ARBT", "FILE", "NEW", idnum, NR, NZ]

        header_str = format_2000.write(case_strings)
        with open(eqdsk_output_path, "w") as file:
            file.write(header_str + "\n")
            # (2020) rdim,zdim,rcentr,rleft,zmid
            file.write(format_2020.write([RDIM, ZDIM, RCENTR, RLEFT, ZMID]) + "\n")
            # (2020) rmaxis,zmaxis,simag,sibry,bcentr
            file.write(format_2020.write([RMAXIS, ZMAXIS, SIMAG, SIBRY, BCENTR]) + "\n")
            # (2020) current,simag,xdum,rmaxis,xdum
            file.write(format_2020.write([CURRENT, SIMAG, XDUM, RMAXIS, XDUM]) + "\n")
            # (2020) zmaxis,xdum,sibry,xdum,xdum
            file.write(format_2020.write([ZMAXIS, XDUM, SIBRY, XDUM, XDUM]) + "\n")
            # (2020) (fpol(i),i=1,nw)
            file.write(format_2020.write(fpol_new) + "\n")
            # (2020) (pres(i),i=1,nw)
            file.write(format_2020.write(pres_new) + "\n")
            # (2020) (ffprim(i),i=1,nw)
            file.write(format_2020.write(ffpr_new) + "\n")
            # (2020) (pprime(i),i=1,nw)
            file.write(format_2020.write(pprime_new) + "\n")
            # (2020) ((psirz(i,j),i=1,nw),j=1,nh)
            towrite = np.array(grid_lin, dtype=float).flatten(order="F")
            file.write(format_2020.write(towrite) + "\n")
            # (2020) (qpsi(i),i=1,nw)
            file.write(format_2020.write(qpsi_new) + "\n")
            # (2022) nbbbs,limitr
            file.write(format_2022.write([NBBBS, limitr]) + "\n")
            # (2020) (rbbbs(i),zbbbs(i),i=1,nbbbs)
            # See: https://github.com/Fusion-Power-Plant-Framework/eqdsk/blob/main/eqdsk/file.py#L733
            towrite = np.array([RBBBS, ZBBBS]).flatten(order="F")
            file.write(format_2020.write(towrite) + "\n")
            # (2020) (rlim(i),zlim(i),i=1,limitr)
            towrite = np.array([RLIM, ZLIM]).flatten(order="F")
            file.write(format_2020.write(towrite) + "\n")

    
    def calc_mixing_length(self, scanfiles_dir, suffix):
        field_path = os.path.join(scanfiles_dir,f'field_{suffix}')
        field = GF(field_path)
        field_dict = field.field_filepath_to_dict(time_criteria='last')
        zgrid = field_dict['zgrid']
        apar = np.abs(field_dict['field_apar'][-1])
        phi = np.abs(field_dict['field_phi'][-1])
        
        cwd = os.getcwd()
        os.chdir(scanfiles_dir)
        pars = init_read_parameters_file('_'+suffix)
        geom_type, geom_pars, geom_coeff = init_read_geometry_file('_'+suffix, pars)
        os.chdir(cwd)
        
        kperp, omd_curv, omd_gradB = calc_kperp_omd(geom_type,geom_coeff,pars,False,False)
        
        avg_kperp_squared_phi = np.sum((phi/np.sum(phi)) * kperp**2)
        avg_kperp_squared_A = np.sum((apar/np.sum(apar)) * kperp**2)
        gamma = pars['kymin']
        return [gamma/avg_kperp_squared_phi, gamma/avg_kperp_squared_A]
    
    def get_all_suffixes(self, scanfiles_dir):
        files = os.listdir(scanfiles_dir)
        pattern = r'\d{4}'
        matched_strings = []
        for filename in files:
            matches = re.findall(pattern, filename)
            matched_strings.extend(matches)
        suffix_s = np.unique(matched_strings)
        return suffix_s
    
    def get_all_mixing_lengths(self, scanfiles_dir):
        suffix_s = self.get_all_suffixes(scanfiles_dir)

        mixing_lengths_phi = []
        mixing_lengths_A = []
        for suffix in suffix_s:
            mxl_phi, mxl_A = self.calc_mixing_length(scanfiles_dir, suffix)
            mixing_lengths_phi.append(mxl_phi)
            mixing_lengths_A.append(mxl_A)
            
        return mixing_lengths_phi, mixing_lengths_A
    
    def read_fluxes(self, scanfiles_dir, suffix, species_names=['Electrons','Ions']):
        nspecies = len(species_names)
        print('READING FLUXES')
        nrg_path = os.path.join(scanfiles_dir,f'nrg_{suffix}')
        with open(nrg_path, 'r') as nrg_file:
            lines = deque(nrg_file, maxlen=nspecies)
            #lines = nrg_file.readlines()[-nspecies:]
        species_fluxes = {}
        for species_name, l in zip(species_names, lines):
            values = re.findall("(-?\d+\.\d+E[+-]?\d+)", l)#np.array(l.split('  ')"
            fluxes = values[4:8]
            fluxes = {f'particle_electrostatic_{species_name}':float(fluxes[0]), f'particle_electromagnetic_{species_name}':float(fluxes[1]), f'heat_electrostatic_{species_name}':float(fluxes[2]), f'heat_electromagnetic_{species_name}':float(fluxes[3])}
            species_fluxes[species_name] = fluxes
        # The species are in the same order as in the gene_parameters file.
        return species_fluxes
    
    def get_fingerprints(self, scanfiles_dir, suffix, species_names=['Electrons','Ions']):
        parameters_path = os.path.join(scanfiles_dir, f'parameters_{suffix}')
        parameters_dict = self.read_parameters_dict(parameters_path)
        nref = parameters_dict['units']['nref']
        Tref = parameters_dict['units']['tref']
        Lref = parameters_dict['units']['lref']
        keys = list(parameters_dict.keys())
        species = [s for s in keys if 'species' in s]
        fluxes = self.read_fluxes(scanfiles_dir, suffix)
        diffusivities = {}
        for name, spec in zip(species_names,species):
            particle_flux = fluxes[name][f'particle_electrostatic_{name}'] + fluxes[name][f'particle_electromagnetic_{name}']#fluxes_df[f'particle_electrostatic_{name}'].iloc[i] + fluxes_df[f'particle_electromagnetic_{name}'].iloc[i]
            omn = parameters_dict[spec]['omn']
            n = nref * parameters_dict[spec]['dens']
            grad_n = -(n/Lref) * omn
            particle_diff = - particle_flux / grad_n
            
            heat_flux = fluxes[name][f'heat_electrostatic_{name}'] + fluxes[name][f'heat_electromagnetic_{name}']            
            omt = parameters_dict[spec]['omt'] 
            T = parameters_dict[spec]['temp'] * Tref
            grad_T = omt * -(T/Lref)
            
            heat_diff = -(heat_flux - (3/2)*T*particle_flux)/(n * grad_T)
            diffusivities[name] = {'particle_diff':particle_diff, 'heat_diff':heat_diff}
        
        fingerprints = [diffusivities['Ions']['heat_diff'] / diffusivities['Electrons']['heat_diff'], diffusivities['Electrons']['particle_diff'] / diffusivities['Electrons']['heat_diff']]
        return fingerprints
    
    def get_local_equlibrium(self, scanfiles_dir, suffix):
        cwd = os.getcwd()
        os.chdir(scanfiles_dir)
        pars = init_read_parameters_file('_'+suffix)
        geom_type, geom_pars, geom_coeff = init_read_geometry_file('_'+suffix, pars)
        os.chdir(cwd)
        # print('GEOME TYPE', geom_type)
        # print('GEOME PARS', geom_pars)
        # print('GEOME COEFF', geom_coeff)
        return geom_pars, geom_coeff
    
    #ANNA: your version returns a dict that only has one species in it.
    def read_parameters_dict(self, parameters_path):
        # for some reason f90nml fails to parse with 'FCVERSION' line in the parameters file, so I comment it
        with open(parameters_path, 'r') as parameters_file:
            lines = parameters_file.readlines()
            for i, line in enumerate(lines):
                if 'FCVERSION' in line:
                    lines[i] = '!'+line

        with open(parameters_path, 'w') as parameters_file:
            parameters_file.writelines(lines)

        with open(parameters_path, 'r') as parameters_file:
            nml = f90nml.read(parameters_file)
            parameters_dict= nml.todict()
        return parameters_dict
    
    def get_normalised_gradients(self, scanfiles_dir, suffix):
        parameters_path = os.path.join(scanfiles_dir, f'parameters_{suffix}')
        params = self.read_parameters_dict(parameters_path)
        norm_grad = [params['_grp_species_0']['omt'], params['_grp_species_0']['omn'], params['_grp_species_1']['omn']] 
        return norm_grad
        
    def get_model_inputs(self, scanfiles_dir, suffix):
        print('GETTING MODEL INPUTS')
        norm_grad = self.get_normalised_gradients(scanfiles_dir, suffix)
        geom_pars, geom_coeff = self.get_local_equlibrium(scanfiles_dir, suffix)
        print('NORM GRAD', norm_grad)
        print('GEOM COEFF', geom_coeff)
        x = norm_grad + [geom_pars['q0'], geom_pars['shat'], geom_pars['s0'], geom_pars['beta'], geom_pars['my_dpdx']]
        
        for k,v in geom_coeff.items():
            x = x + list(v)
        
        return np.array(x)
    
    def get_model_outputs(self, scanfiles_dir, suffix):
        mixing_lengths = self.calc_mixing_length(scanfiles_dir, suffix)
        fingerprints = self.get_fingerprints(scanfiles_dir,suffix)
        y = mixing_lengths + fingerprints
        return np.array(y)
    
    def write_parameters_file_pyro(self, psi_n, iterdb_path, eqdsk_path, write_path):
        from pyrokinetics import Pyro
        # Load up pyro object
        pyro = Pyro(
            eq_file=eqdsk_path,
            eq_type="GEQDSK",
            gk_code="GENE",
        )
        from pyrokinetics_extra.iterdb import KineticsReaderITERDB
        reader = KineticsReaderITERDB()
        kinetics = reader.read_from_file(iterdb_path, pyro.eq)
        pyro.kinetics = kinetics
        pyro.load_local_geometry(psi_n=psi_n, local_geometry="FourierGENE", show_fit=False)
        pyro.write_gk_file(file_name=write_path)
        
        