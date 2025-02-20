#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from fieldlib import *
from ParIO import *
import sys
from get_nrg import get_nrg0
from calc_gr import *
from finite_differences import *
from read_write_geometry import *
import optparse as op
from subprocess import call
from get_abs_psi_prime import *
from remove_edge_opt import *

parser=op.OptionParser(description='Calculates mode information and synthesizes scan info.')
options,args=parser.parse_args()
if len(args)!=1:
    exit("""
Please include scan number as argument (e.g., 0001)."
    \n""")
suffix = args[0]

calc_grs = True
os.chdir('scanfiles'+suffix)


def my_corr_func_complex(v1,v2,time,show_plot=False,v1eqv2=True):
    #print( "len(time)",len(time))
    #print "len(v1)",len(v1)
    #print "len(v2)",len(v2)
    dt=time[1]-time[0]
    #print "dt:", dt
    N=len(time)
    cfunc=np.zeros(N,dtype='complex')
    for i in range(N):
        i0=i+1
        cfunc[-i0]=np.sum(np.conj(v1[-i0:])*v2[:i0])
    tau=np.arange(N)
    tau=tau*dt
    if v1eqv2:
        cfunc=np.real(cfunc)
    max_corr=max(np.abs(cfunc))
    corr_time=0.0
    i=0
    while corr_time==0.0:
        if (abs(cfunc[i])-max_corr/np.e) > 0.0 and \
           (abs(cfunc[i+1])-max_corr/np.e) <= 0.0:
            slope=(cfunc[i+1]-cfunc[i])/(tau[i+1]-tau[i])
            zero=cfunc[i]-slope*tau[i]
            corr_time=(max_corr/np.e-zero)/slope
        i+=1
    neg_loc = 10000.0
    i=0
    while neg_loc==10000.0 and i < N:
        if cfunc[i] < 0.0:
            neg_loc = tau[i]
        i+=1

    if neg_loc < corr_time:
        print( "WARNING: neg_loc < corr_time")
        corr_time = neg_loc

    if show_plot:
        plt.plot(tau,cfunc,'x-')
        ax=plt.axis()
        plt.vlines(corr_time,ax[2],ax[3])
        plt.show()
    return cfunc,tau,corr_time








par = Parameters()
par.Read_Pars('parameters')
pars = par.pardict
if 'edge_opt' in pars:
    edge_opt = pars['edge_opt']
else:
    edge_opt = 1
print( "edge_opt = ",edge_opt)
if pars['n_spec'] == 3:
   print( "Species 3:",pars['name3']   )
#   dummy = raw_input("Warning!  Assuming species 3 is electrons (press any key):\n")

print( type(pars['scan_dims']))
print( "scan_dims",pars['scan_dims'])
if type(pars['scan_dims']) == str:
    scan_dims = pars['scan_dims'].split()
    numscan_tot = 1
    for i in scan_dims:
        numscan_tot *= int(i)
else:
    numscan_tot = pars['scan_dims']

print( "Total number of runs: ", numscan_tot )






#Test if global scan
if 'x_local' in pars and not pars['x_local']:
    scan_info = np.zeros((numscan_tot,20),dtype='float64')
    for i in range(numscan_tot):
        par0 = Parameters()
        scan_num = '000'+str(i+1)
        scan_num = scan_num[-4:]
        print( "Analyzing ",scan_num)
        if os.path.isfile('parameters_'+scan_num):
            par0.Read_Pars('parameters_'+scan_num)
            pars0 = par0.pardict
            nspec = pars0['n_spec']
            print( pars0['kymin'])
            scan_info[i,0] = pars0['kymin']
            if 'x0' in pars0:
                scan_info[i,1] = pars0['x0']
            elif 'x0' in pars:
                scan_info[i,1] = pars['x0']
            else:
                scan_info[i,1] = np.nan
            scan_info[i,2] = 0.0
            if 'n0_global' in pars0:
                scan_info[i,3] = pars0['n0_global']
            else:
                scan_info[i,3] = np.nan
        else:
            par0.Read_Pars('in_par/parameters_'+scan_num)
            pars0 = par0.pardict
            nspec = pars0['n_spec']
            scan_info[i,0] = float(str(pars0['kymin']).split()[0])
            if 'n0_global' in pars0:
                scan_info[i,3] = float(str(pars0['n0_global']).split()[0])
            else:
                scan_info[i,3] = np.nan
            scan_info[i,1] = pars0['x0']
            scan_info[i,2] = 0.0
        if os.path.isfile('omega_'+scan_num):
            omega0 = np.genfromtxt('omega_'+scan_num)
            if omega0.any() and omega0[1] != 0.0:
                scan_info[i,4]=omega0[1]
                scan_info[i,5]=omega0[2]
            elif calc_grs:
                call(['calc_omega_from_field.py',scan_num])
                om = np.genfromtxt('omega_'+scan_num)
                scan_info[i,4]=om[1]
                scan_info[i,5]=om[2]
            else:
                scan_info[i,4]=np.nan
                scan_info[i,5]=np.nan
        elif calc_grs and os.path.isfile('field_'+scan_num):
            call(['calc_omega_from_field.py',scan_num])
            om = np.genfromtxt('omega_'+scan_num)
            scan_info[i,4]=om[1]
            scan_info[i,5]=om[2]
        else:
            scan_info[i,4]=np.nan
            scan_info[i,5]=np.nan
        if os.path.isfile('field_'+scan_num):
            field = fieldfile('field_'+scan_num,pars0)
            field.set_time(field.tfld[-1])
            #field.set_time(field.tfld[-1],len(field.tfld)-1)
            fntot = field.nz*field.nx
    
            dz = float(2.0)/float(field.nz)
            zgrid = np.arange(field.nz)/float(field.nz-1)*(2.0-dz)-1.0
            zgrid_ext = np.arange(field.nz+4)/float(field.nz+4-1)*(2.0+3*dz)-(1.0+2.0*dz)
            field.set_time(field.tfld[-1])
            #field.set_time(field.tfld[-1],len(field.tfld)-1)
       
            imax = np.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(field.nz,field.nx))
            imaxa = np.unravel_index(np.argmax(abs(field.apar()[:,0,:])),(field.nz,field.nx))
            phi = field.phi()[:,0,:]
            apar = field.apar()[:,0,:] 
            cfunc,zed,corr_len=my_corr_func_complex(phi[:,imax[1]],phi[:,imax[1]],zgrid,show_plot=False)
            scan_info[i,6] = np.nan
            scan_info[i,7] = corr_len
            parity_factor_apar = np.abs(np.sum(apar[:,imaxa[1]]))/np.sum(np.abs(apar[:,imaxa[1]]))
            scan_info[i,8] = parity_factor_apar
            parity_factor_phi = np.abs(np.sum(phi[:,imax[1]]))/np.sum(np.abs(phi[:,imax[1]]))
            scan_info[i,9] = parity_factor_phi
    
            #KBM test with E||

            dz = 2.0/field.nz
            zgrid_ext = np.arange(field.nz+4)/float(field.nz+4-1)*(2.0+3*dz)-(1.0+2.0*dz)
            xgrid = np.arange(field.nx)/float(field.nx-1)*pars0['lx_a']+pars0['x0']-pars0['lx_a']/2.0
            gpars,geometry = read_geometry_global(pars0['magn_geometry']+'_'+scan_num)
            phase = (0.0+1.0J)*pars0['n0_global']*2.0*np.pi*geometry['q']
            phi_bnd = np.zeros((field.nz+4,field.nx),dtype = 'complex128')     
            gradphi= np.zeros((field.nz+4,field.nx),dtype = 'complex128')     
            phi_bnd[2:-2,:] = phi
            for j in range(field.nx):
                phi_bnd[-2,j] = phi_bnd[2,j]*np.e**(-phase[j])
                phi_bnd[-1,j] = phi_bnd[3,j]*np.e**(-phase[j])
                phi_bnd[0,j] = phi_bnd[-4,j]*np.e**(phase[j])
                phi_bnd[1,j] = phi_bnd[-3,j]*np.e**(phase[j])
                gradphi[:,j] = fd_d1_o4(phi_bnd[:,j],zgrid_ext)
                gradphi[2:-2,j] = gradphi[2:-2,j]/np.pi/(geometry['jacobian'][:,j]*geometry['Bfield'][:,j])
        
            field.set_time(field.tfld[-1])
            #field.set_time(field.tfld[-1],len(field.tfld)-1)
        
            #apar = field.apar()[:,:]
        
            omega_complex = (scan_info[i,5]*(0.0+1.0J) + scan_info[i,4])
        
            print("omega_complex",omega_complex)
            
            diff = np.sum(np.abs(gradphi[2:-2,:] + omega_complex*apar[:,:]))
            phi_cont = np.sum(np.abs(gradphi[2:-2,:]))
            apar_cont = np.sum(np.abs(omega_complex*apar[:,:]))
            print( "diff",diff)
            print( "phi_cont",phi_cont)
            print( "apar_cont",apar_cont)
            print( "diff/abs",diff/(phi_cont+apar_cont))
            scan_info[i,11] = diff/(phi_cont+apar_cont)
            scan_info[i,12] = np.nan           
            scan_info[i,13] = np.nan
   
        else:
            scan_info[i,6] = np.nan
            scan_info[i,7] = np.nan
            scan_info[i,8] = np.nan
            scan_info[i,9] = np.nan
            scan_info[i,11] = np.nan
            scan_info[i,12] = np.nan
            scan_info[i,13] = np.nan
    
        if os.path.isfile('nrg_'+scan_num):
            if nspec==1:
                tn,nrg1=get_nrg0('_'+scan_num,nspec=nspec)
                scan_info[i,10]=nrg1[-1,7]/abs(nrg1[-1,6])
            elif nspec==2:
                tn,nrg1,nrg2=get_nrg0('_'+scan_num,nspec=nspec)
                scan_info[i,10]=nrg2[-1,7]/(abs(nrg2[-1,6])+abs(nrg1[-1,6]))
            elif nspec==3:
                tn,nrg1,nrg2,nrg3=get_nrg0('_'+scan_num,nspec=nspec)
                scan_info[i,10]=nrg2[-1,7]/(abs(nrg2[-1,6])+abs(nrg1[-1,6]))
            else:
                sys.exit("Not ready for nspec>3")
        else:
            scan_info[i,10] = np.nan
    
    f=open('scan_info.dat','w')
    p = open('parameters','r')
    lines = p.read()
    p.close()
    lines = lines.split('\n')
    for line in lines:
        if 'scan' in line:
            f.write('#'+line+'\n')
    f.write('#1.kymin 2.x0 3.kx_center 4.n0_global 5.gamma(cs/a) 6.omega(cs/a) 7.<z> 8.lambda_z 9.parity(apar) 10.parity(phi) 11.QEM/QES 12.Epar cancelation \n')
    np.savetxt(f,scan_info)
    f.close()






#local scan
else:

    scan_info = np.zeros((numscan_tot,27),dtype='float64')
    
    for i in range(numscan_tot):
        par0 = Parameters()
        scan_num = '000'+str(i+1)
        scan_num = scan_num[-4:]
        print( "Analyzing ",scan_num)
        if os.path.isfile('parameters_'+scan_num):
            par0.Read_Pars('parameters_'+scan_num)
            pars0 = par0.pardict
            nspec = pars0['n_spec']
            print( pars0['kymin'])
            scan_info[i,0] = pars0['kymin']
            if 'x0' in pars0:
                scan_info[i,1] = pars0['x0']
            elif 'x0' in pars:
                scan_info[i,1] = pars['x0']
            else:
                scan_info[i,1] = np.nan
            if 'kx_center' in pars0:
                scan_info[i,2] = pars0['kx_center']
            else:
                scan_info[i,2] = 0.0
            if 'n0_global' in pars0:
                scan_info[i,3] = pars0['n0_global']
            else:
                scan_info[i,3] = np.nan
        else:
            par0.Read_Pars('in_par/parameters_'+scan_num)
            pars0 = par0.pardict
            nspec = pars0['n_spec']
            scan_info[i,0] = float(str(pars0['kymin']).split()[0])
            if 'x0' in pars0:
                scan_info[i,1] = float(str(pars0['x0']).split()[0])
            else:
                scan_info[i,1] = np.nan
            if 'kx_center' in pars0:
                scan_info[i,2] = float(str(pars0['kx_center']).split()[0])
            else:
                scan_info[i,2] = 0.0
            if 'n0_global' in pars0:
                scan_info[i,3] = pars0['n0_global']
            else:
                scan_info[i,3] = np.nan
        if os.path.isfile('omega_'+scan_num):
            omega0 = np.genfromtxt('omega_'+scan_num)
            if omega0.any() and omega0[1] != 0.0:
                scan_info[i,4]=omega0[1]
                scan_info[i,5]=omega0[2]
            elif calc_grs:
                scan_info[i,4]=calc_gr2('_'+scan_num,nspec=nspec)
                scan_info[i,5]= 0.0
                #np.savetxt('omega_'+scan_num,[scan_info[i,0],scan_info[i,4],np.nan])
                f=open('omega_'+scan_num,'w')
                f.write(str(pars0['kymin'])+'    '+str(scan_info[i,4])+'    '+str(scan_info[i,5])+'\n')
                f.close()
            else:
                scan_info[i,4]=np.nan
                scan_info[i,5]=np.nan
        elif calc_grs and os.path.isfile('nrg_'+scan_num):
            scan_info[i,4]=calc_gr2('_'+scan_num,nspec=nspec)
            scan_info[i,5] = 0.0
            np.savetxt('omega_'+scan_num,[scan_info[i,0],scan_info[i,4],np.nan])
        else:
            scan_info[i,4]=np.nan
            scan_info[i,5]=np.nan
        
        if os.path.isfile('field_'+scan_num):
            field = fieldfile('field_'+scan_num,pars0)
            #field.set_time(field.tfld[-1],len(field.tfld)-1)
            field.set_time(field.tfld[-1])
            fntot = field.nz*field.nx
    
            dz = float(2.0)/float(field.nz)
            zgrid = np.arange(fntot)/float(fntot-1)*(2*field.nx-dz)-field.nx
            zgrid0 = np.arange(field.nz)/float(field.nz-1)*(2.0-dz)-1.0
            phi = np.zeros(fntot,dtype='complex128')
            apar = np.zeros(fntot,dtype='complex128')
            phikx = field.phi()[:,0,:]
            aparkx = field.apar()[:,0,:]
            if 'n0_global' in pars0:
                phase_fac = -np.e**(-2.0*np.pi*(0.0+1.0J)*pars0['n0_global']*pars0['q0'])
            else:
                #print "pars0['shat']",pars0['shat']
                #print "pars0['kymin']",pars0['kymin']
                #print "pars0['lx']",pars0['lx']
                phase_fac = -np.e**(-np.pi*(0.0+1.0J)* pars0['shat']*pars0['kymin']*pars0['lx'])
            for j in range(int(field.nx/2)):
                phi[(j+int(field.nx/2))*field.nz:(j+int(field.nx/2)+1)*field.nz]=field.phi()[:,0,j]*phase_fac**j
                if j < field.nx/2:
                    phi[(int(field.nx/2)-j-1)*field.nz : (int(field.nx/2)-j)*field.nz ]=field.phi()[:,0,-1-j]*phase_fac**(-(j+1))
                if pars0['n_fields']>1:
                    apar[(j+int(field.nx/2))*field.nz:(j+int(field.nx/2)+1)*field.nz]=field.apar()[:,0,j]*phase_fac**j
                    if j < field.nx/2:
                        apar[(int(field.nx/2)-j-1)*field.nz : (int(field.nx/2)-j)*field.nz ]=field.apar()[:,0,-1-j]*phase_fac**(-(j+1))
        
            zavg=np.sum(np.abs(phi)*np.abs(zgrid))/np.sum(np.abs(phi))
            scan_info[i,6] = zavg
            cfunc,zed,corr_len=my_corr_func_complex(phi,phi,zgrid,show_plot=False)
            scan_info[i,7] = corr_len
            parity_factor_apar = np.abs(np.sum(apar))/np.sum(np.abs(apar))
            scan_info[i,8] = parity_factor_apar
            parity_factor_phi = np.abs(np.sum(phi))/np.sum(np.abs(phi))
            scan_info[i,9] = parity_factor_phi
    
            #KBM test with E||
            gpars,geometry = read_geometry_local(pars0['magn_geometry']+'_'+scan_num)
            jacxB = geometry['gjacobian']*geometry['gBfield']
            if scan_info[i,5] == scan_info[i,5]:
                omega_complex = (scan_info[i,5]*(0.0+1.0J) + scan_info[i,4])
                gradphi = fd_d1_o4(phi,zgrid)
                for j in range(pars0['nx0']):
                    gradphi[pars0['nz0']*j:pars0['nz0']*(j+1)] = gradphi[pars0['nz0']*j:pars0['nz0']*(j+1)]/jacxB[:]/np.pi
                #plt.plot(gradphi)
                #plt.plot(omega_complex*apar)
                #plt.show()
                diff = np.sum(np.abs(gradphi + omega_complex*apar))
                phi_cont = np.sum(np.abs(gradphi))
                apar_cont = np.sum(np.abs(omega_complex*apar))
                scan_info[i,11] = diff/(phi_cont+apar_cont)
                #print 'diff/abs',scan_info[i,10]
            else:
                scan_info[i,11] = np.nan
            phi0 = np.empty(np.shape(phikx),dtype = 'complex') 
            apar0 = np.empty(np.shape(aparkx),dtype = 'complex') 
            #print "Shape of phikx",np.shape(phikx)
            #print "Shape of zgrid",np.shape(zgrid)
            #dummy = raw_input("Press any key:\n")
            #if edge_opt > 0.0: 
            #    for ix in range(len(phikx[0,:])):
            #        phi0[:,ix] = remove_edge_opt_complex(phikx[:,ix],edge_opt)
            #        apar0[:,ix] = remove_edge_opt_complex(aparkx[:,ix],edge_opt)
            phi0 = phikx
            apar0 = aparkx
            #Calculate <gamma_HB> / gamma
            geomfile = pars0['magn_geometry']+'_'+scan_num
            print( "geomfile",geomfile)

            #Estimate of global relevance
            dkx = 2.0*np.pi*pars0['shat']*pars0['kymin']
            phi2tot = np.sum(np.sum(abs(phikx)**2,axis=0),axis=0)
            kxgrid = np.empty(pars0['nx0'],dtype='float64')
            if 'kx_center' in pars0:
               kxgrid[0] = pars0['kx_center']
            else:
               kxgrid[0] = 0.0  
            for k in range(int(pars0['nx0']/2)):
               kxgrid[k+1] = kxgrid[0] + (k+1)*dkx  
            for k in range(int((pars0['nx0']-1)/2)):
               kxgrid[-k-1] = kxgrid[0] - (k+1)*dkx  
            print( "kxgrid",kxgrid)
            kxavg = 0.0
            #Get eigenmode averaged |kx|
            for k in range(pars0['nx0']):
               kxavg += abs(kxgrid[k])*np.sum(abs(phikx[:,k])**2)
            kxavg = kxavg/phi2tot
            #global factor is kxavg / (rho/L_{T,n})--i.e. rho/L_eigenmode / (rho / L_gradients) ~ L_gradients / L_eigenmode
            #==> if L_gradients / L_eigenmode is large then the mode can survive in global 
            #Note: filter_local_modes in plot_scan_info_efit.py
            if 'rhostar' in pars0:
               global_factor = kxavg/(pars0['rhostar']*0.5*(pars0['omn1']+pars0['omt1']))
               scan_info[i,13] = global_factor
            else:
               global_factor = 0
               scan_info[i,13] = global_factor
            call(['calc_kpar_kperp_omd.py',scan_num]) 
            #call(['calc_kpar_kperp_omd_apar.py',scan_num]) 
            call(['calc_kpar_kperp_omd_new.py','-f 2',scan_num]) 
            data_phi = np.genfromtxt('mode_info_1_'+scan_num)
            #data_apar = np.genfromtxt('mode_info_apar_1_'+scan_num)
            data_apar = np.genfromtxt('mode_info_upar_e_1_'+scan_num)
            scan_info[i,14] = data_phi[1]  #kpar(phi)
            scan_info[i,15] = data_apar[1]  #kpar(apar)
            scan_info[i,16] = data_phi[2]**2  #kperp**2(phi)
            scan_info[i,17] = data_apar[2]**2  #kperp**2(apar)
            scan_info[i,20] = data_phi[7]   #Q/n^2
        else:
            scan_info[i,6] = np.nan
            scan_info[i,7] = np.nan
            scan_info[i,8] = np.nan
            scan_info[i,9] = np.nan
            scan_info[i,11] = np.nan
            scan_info[i,12] = np.nan
            scan_info[i,13] = np.nan
    
        if os.path.isfile('nrg_'+scan_num):
            for i0 in range(3):
              if 'name'+str(i0+1) in pars:
                 print( "i0",i0,str(i0+1),pars['name'+str(i0+1)][1:-1])
                 if pars['name'+str(i0+1)] == 'e':
                    especnum = i0+1
                 if pars['name'+str(i0+1)] == 'i':
                    ispecnum = i0+1
            if nspec==1:
                tn,nrg1=get_nrg0('_'+scan_num,nspec=nspec)
                scan_info[i,10]=nrg1[-1,7]/abs(nrg1[-1,6])
            elif nspec==2:
                tn,nrg1,nrg2=get_nrg0('_'+scan_num,nspec=nspec)
                scan_info[i,10]=nrg2[-1,7]/(abs(nrg2[-1,6])+abs(nrg1[-1,6]))
                scan_info[i,18]=nrg1[-1,6]/abs(nrg2[-1,6])  #QiES/QeES
                scan_info[i,19]=nrg2[-1,4]/abs(nrg2[-1,6])  #GeES/QeES
            elif nspec==3:
                tn,nrg1,nrg2,nrg3=get_nrg0('_'+scan_num,nspec=nspec)
                if ispecnum != 1:
                   print( "Error: main ions must be first species.")
                if especnum == 2:
                   scan_info[i,10]=nrg2[-1,7]/(abs(nrg2[-1,6])+abs(nrg1[-1,6]))
                   scan_info[i,18]=nrg1[-1,6]/abs(nrg2[-1,6])  #QiES/QeES
                   scan_info[i,19]=nrg2[-1,4]/abs(nrg2[-1,6])  #GeES/QeES
                   print("!!!",scan_info[i,10])
                elif especnum == 3:
                   scan_info[i,10]=nrg3[-1,7]/(abs(nrg3[-1,6])+abs(nrg1[-1,6]))
                   print( "Error: Electrons must be second species!!")
                   stop
                else:
                   stop
            else:
                sys.exit("Not ready for nspec>3")
        else:
            scan_info[i,10] = np.nan
        if 'omt1' in pars0 and 'omn1' in pars0:
            scan_info[i,21] = pars0['omt1']
            scan_info[i,22] = pars0['omn1']
        else:
            scan_info[i,21] = 0.0
            scan_info[i,22] = 0.0
        if nspec > 1 and 'omt2' in pars0 and 'omn2' in pars0:
            scan_info[i,23] = pars0['omt2']
            scan_info[i,24] = pars0['omn2']
        else:
            scan_info[i,23] = 0.0
            scan_info[i,24] = 0.0
        if nspec > 2 and 'omt3' in pars0 and 'omn3' in pars0:
            scan_info[i,25] = pars0['omt3']
            scan_info[i,26] = pars0['omn3']
        else:
            scan_info[i,25] = 0.0
            scan_info[i,26] = 0.0

    
    f=open('scan_info.dat','w')
    p = open('parameters','r')
    lines = p.read()
    p.close()
    lines = lines.split('\n')
    for line in lines:
        if 'scan' in line:
            f.write('#'+line+'\n')
    f.write('#1.kymin 2.x0 3.kx_center 4.n0_global 5.gamma(cs/a) 6.omega(cs/a) 7.<z> 8.lambda_z 9.parity(apar) 10.parity(phi) 11.QEM/QES 12.Epar cancelation 13.gamma_HB_avg 14.global_factor 15.kpar(phi) 16.kpar(apar) 17.kperp^2(phi) 18.kperp^2(apar) 19.QiES/QeES 20.GammaeES/QeES 21.Q/n^2 22.omt1 23.omn1 24.omt2 25.omn2 26.omt3 27.omn3\n')
    np.savetxt(f,scan_info)
    f.close()









x0_list = np.empty(0)
for i in range(len(scan_info[:,0])):
    if scan_info[i,1] not in x0_list:
        x0_list = np.append(x0_list,scan_info[i,1])
nx0 = len(x0_list)
gamma_out = np.zeros(nx0)
kperp2_out = np.zeros(nx0)
Qn2_out = np.zeros(nx0)
Qioe_out = np.zeros(nx0)
gamma_over_kperp2_out = np.zeros(nx0)
for i in range(len(scan_info[:,0])):
    ix0 = np.argmin(abs(scan_info[i,2]-x0_list))
    this_gamma = scan_info[i,4]
    this_kperp2 = scan_info[i,16]
    if this_gamma/this_kperp2 > gamma_over_kperp2_out[ix0]:
        gamma_over_kperp2_out[ix0] = this_gamma/this_kperp2
        Qn2_out[ix0] = scan_info[i,20]
        Qioe_out[ix0] = scan_info[i,18]
        gamma_out[ix0] = scan_info[i,4]
        kperp2_out[ix0] = scan_info[i,16]

#f = open('mixing_length_info.dat','w')
#f.write('#1.x0 2.gamma 3.kperp2 4.Q/n^2 5.Qi/Qe 6.gamma/kperp2\n')
#f.write(np.column_stack((x0_list,gamma_out,kperp2_out,Qn2_out,Qioe_out,gamma_over_kperp2_out)))
#f.close()
#
    






    

