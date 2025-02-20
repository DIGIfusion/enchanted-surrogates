#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
from fieldlib import *
from ParIO import * 
from finite_differences import *
import optparse as op
from read_write_geometry import *
from subprocess import call
import sys
import math
from interp import *
from read_iterdb import *
#from calc_omega_from_field import *

parser=op.OptionParser(description='Plots mode structures and calculates various interesting quantities.')
parser.add_option('--plot_theta','-g',action='store_const',const=False,help = 'Plot global mode structures decomposed in poloidal m number.',default=True)
parser.add_option('--calc_epar','-e',action='store_true',help = 'Calculate E_parallel.',default=False)
parser.add_option('--plot_ballooning','-b',action='store_const',const=False,help = 'Plot global mode structures in z, r.',default=True)
#parser.add_option('--plot_all','-a',action='store_const',const=1,help = 'Plot all plots.',default=False)
parser.add_option('--time','-t',type = 'float',action='store',dest="time0",help = 'Time to plot mode structure.',default=-1)
parser.add_option('--idb','-i',type = 'str',action='store',dest="idb_file",help = 'ITERDB file name.',default='empty')
parser.add_option('--pfile','-f',type = 'str',action='store',dest="prof_file",help = 'ITERDB file name.',default='empty')
parser.add_option('--output','-o',action='store_true',help = 'Output eigenmode structure in ascii file.',default=False)
options,args=parser.parse_args()
print("options",options)
print("args",args)
if len(args)!=1:
    exit("""
Please include run number as argument (e.g., 0001)."
    \n""")
suffix = args[0]

if suffix in ['dat','.dat']:
     suffix = '.dat'
else:
     suffix = '_'+suffix

#plot_all=options.plot_all
output_file=options.output
print("options.plot_ballooning", options.plot_ballooning)
plot_ballooning=options.plot_ballooning
plot_theta=options.plot_theta
calc_epar=options.calc_epar
print("calc_epar",calc_epar)
idb_file = options.idb_file
prof_file = options.prof_file
time0=float(options.time0)

par = Parameters()
par.Read_Pars('parameters'+suffix)
pars = par.pardict

#field = fieldfile('field'+suffix,pars,False)
field = fieldfile('field'+suffix,pars)
#print "time0",time0
#print "field.tfld",field.tfld
time = np.array(field.tfld)
if time0 == -1:
    itime = -1
    itime0 = len(time)-1
else: 
    itime = np.argmin(abs(time - time0))
    itime0 = itime

print("Looking at the mode structure at time:",time[itime])
#field.set_time(time[itime],itime0)
field.set_time(time[itime])

ntot = field.nz*field.nx

def my_corr_func_complex(v1,v2,time,show_plot=False,v1eqv2=True):
    #print "len(time)",len(time)
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
    neg_loc = 0.0
    i=0
    while neg_loc==0.0 and i < N:
        if cfunc[i] < 0.0:
            neg_loc = tau[i]
        i+=1

    if neg_loc < corr_time:
        print("WARNING: neg_loc < corr_time")
        corr_time = neg_loc

    if show_plot:
        plt.plot(tau,cfunc,'x-')
        ax=plt.axis()
        plt.vlines(corr_time,ax[2],ax[3])
        plt.show()
    return cfunc,tau,corr_time

dz = float(2*field.nx)/ntot
#print 'dz',dz
dz = float(2.0)/float(field.nz)
#print 'dz',dz
zgrid = np.arange(ntot)/float(ntot-1)*(2*field.nx-dz)-field.nx
#print 'zgrid',zgrid

phi = np.zeros(ntot,dtype='complex128')
apar = np.zeros(ntot,dtype='complex128')
#print "ntot",field.nz*field.nx

if 'x_local' in pars:
    if pars['x_local']:
        x_local = True
    else:
        x_local = False 
else:
    x_local = True

if x_local:

    if 'n0_global' in pars:
        phase_fac = -np.e**(-2.0*np.pi*(0.0+1.0J)*pars['n0_global']*pars['q0'])
    else:
        phase_fac = -1.0
    #print "phase_fac",phase_fac

    if pars['shat'] < 0.0:
        for i in range(int(field.nx/2)+1):
            print("phase_fac**i",phase_fac**i)
            print("phase_fac**(-(i+1))",phase_fac**(-(i+1)))
            phi[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.phi()[:,0,-i]*phase_fac**i
            if i < int(field.nx/2):
                phi[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.phi()[:,0,i+1]*phase_fac**(-(i+1))
            if pars['n_fields']>1:
                apar[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.apar()[:,0,-i]*phase_fac**i
                if i < int(field.nx/2):
                    apar[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.apar()[:,0,i+1]*phase_fac**(-(i+1))
    else:
        for i in range(int(field.nx/2)):
            print("phase_fac**i",phase_fac**i)
            print("phase_fac**(-(i+1))",phase_fac**(-(i+1)))
            #print phi[(i+field.nx/2)*field.nz:(i+field.nx/2+1)*field.nz]
            #print (i+field.nx/2)*field.nz
            #print (i+field.nx/2+1)*field.nz
            phi[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.phi()[:,0,i]*phase_fac**i
            if i < int(field.nx/2):
                phi[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.phi()[:,0,-1-i]*phase_fac**(-(i+1))
            if pars['n_fields']>1:
                apar[(i+int(field.nx/2))*field.nz:(i+int(field.nx/2)+1)*field.nz]=field.apar()[:,0,i]*phase_fac**i
                if i < int(field.nx/2):
                    apar[(int(field.nx/2)-i-1)*field.nz : (int(field.nx/2)-i)*field.nz ]=field.apar()[:,0,-1-i]*phase_fac**(-(i+1))
    
    zavg=np.sum(np.abs(phi)*np.abs(zgrid))/np.sum(np.abs(phi))
    print("zavg (for phi)",zavg)
    phi = phi/field.phi()[int(field.nz/2),0,0]
    apar = apar/field.phi()[int(field.nz/2),0,0]
    #print "phi",phi
    #print "apar",apar
    #phi[(field.nx/2)*field.nz:(field.nx/2+1)*field.nz]=field.phi()[:,0,0]
    #for i in range(1,field.nx/2+1):
    #    phi[(i+field.nx/2)*field.nz:(i+field.nx/2+1)*field.nz] = field.phi()[:,0,-1-(i-1)]*(-1)**(i)
    
    if output_file:
        phi = phi / np.max(np.real(phi))
        apar = apar / np.max(np.real(apar))
        f = open('mode_structure'+suffix,'w')
        np.savetxt(f,np.column_stack((zgrid,np.real(phi),np.imag(phi),np.real(apar),np.imag(apar))))
        f.close()
    elif plot_ballooning:
        plt.title(r'$\phi$')
        plt.plot(zgrid,np.real(phi),color='red',label=r'$Re[\phi]$')
        plt.plot(zgrid,np.imag(phi),color='blue',label=r'$Im[\phi]$')
        plt.plot(zgrid,np.abs(phi),color='black',label=r'$|\phi|$')
        ax=plt.axis()
        plt.axvline(zavg,ax[2],ax[3],color='yellow')
        plt.axvline(-zavg,ax[2],ax[3],color='yellow')
        plt.legend()
        plt.xlabel(r'$z/\pi$',size=18)
        plt.show()
    
        plt.title(r'$A_{||}$')
        plt.plot(zgrid,np.real(apar),color='red',label=r'$Re[A_{||}]$')
        plt.plot(zgrid,np.imag(apar),color='blue',label=r'$Im[A_{||}]$')
        plt.plot(zgrid,np.abs(apar),color='black',label=r'$|A_{||}|$')
        plt.legend()
        plt.xlabel(r'$z/\pi$',size=18)
        plt.show()
        #np.savetxt('apar'+suffix,np.column_stack((zgrid,np.real(apar),np.imag(apar))))
    
    cfunc,zed,corr_len=my_corr_func_complex(phi,phi,zgrid,show_plot=False)
    print("correlation length (for phi): ", corr_len)
    parity_factor_apar = np.abs(np.sum(apar))/np.sum(np.abs(apar))
    print("parity factor (for apar):",parity_factor_apar)
    parity_factor_phi = np.abs(np.sum(phi))/np.sum(np.abs(phi))
    print("parity factor (for phi):",parity_factor_phi)
    apar_phase = np.sum(abs(np.real(apar)+np.imag(apar)))/np.sum(abs(np.real(apar))+abs(np.imag(apar)))
    print("Phase factor (apar):",apar_phase)
    
    #Estimate E_parallel
    #phiR = np.real(phi)
    #phiI = np.imag(phi)
    #aparR = np.real(apar)
    #aparI = np.imag(apar)
    #gradphiR = fd_d1_o4(phiR,zgrid)
    #gradphiI = fd_d1_o4(phiI,zgrid)
    if calc_epar:
       print("Hello")
       if os.path.isfile('omega'+suffix):
           om = np.genfromtxt('omega'+suffix)
       else:
           #call(['calc_omega_from_field.py',suffix[1:],'-a'])
           call(['calc_omega_from_field.py',suffix[1:]])
           om = np.genfromtxt('omega'+suffix)

       if om.any():
           if om[0] == 0.0 or om[1] == 0.0 or om[1] != om[1]:
               #call(['calc_omega_from_field.py',suffix[1:],'-a'])
               call(['calc_omega_from_field.py',suffix[1:]])
               om = np.genfromtxt('omega'+suffix)
       else:
           #call(['calc_omega_from_field.py',suffix[1:],'-a'])
           call(['calc_omega_from_field.py',suffix[1:]])
           om = np.genfromtxt('omega'+suffix)

    #omega_apar = (om[2]+om[1]*(0.0+1.0J))*apar
    #omega_apar = (om[2])*apar
    #plt.plot(zgrid,gradphiR,label='gradphiR')
    #plt.plot(zgrid,gradphiI,label='gradphiI')
    #plt.plot(zgrid,np.real(omega_apar),label='omega_aparR')
    #plt.plot(zgrid,np.imag(omega_apar),label='omega_aparI')
    #plt.legend()
    #plt.show()
    
    #Note:  the complex frequency is (gamma + i*omega)
    #print "Here"
    gpars,geometry = read_geometry_local(pars['magn_geometry']+suffix)
    jacxB = geometry['gjacobian']*geometry['gBfield']
    #plt.plot(geometry['gjacobian']*geometry['gBfield'])
    #plt.title('Jac x B')
    #plt.show()
    omega_complex = (om[2]*(0.0+1.0J) + om[1])
    omega_phase = np.log(omega_complex/np.abs(omega_complex))/(0.0+1.0J)
    print("omega_complex",omega_complex)
    print("omega_phase",omega_phase)
    phase_array = np.empty(len(zgrid))
    phase_array[:] = np.real(omega_phase)

    gradphi = fd_d1_o4(phi,zgrid)
    for i in range(pars['nx0']):
        gradphi[pars['nz0']*i:pars['nz0']*(i+1)] = gradphi[pars['nz0']*i:pars['nz0']*(i+1)]/jacxB[:]/np.pi
    
       #ratio = gradphi/-apar
       #mode_phase = np.log(ratio / np.abs(ratio))/(0.0+1.0J)
    #plt.plot(zgrid,mode_phase)
    #plt.plot(zgrid,phase_array,'-.',color = 'black')
    #plt.show()
        
    diff = np.sum(np.abs(gradphi + omega_complex*apar))
    phi_cont = np.sum(np.abs(gradphi))
    apar_cont = np.sum(np.abs(omega_complex*apar))
    print("diff",diff)
    print("phi_cont",phi_cont)
    print("apar_cont",apar_cont)
    print("diff/abs",diff/(phi_cont+apar_cont))
    
    if (not output_file) and plot_ballooning:
        plt.plot(zgrid,np.real(gradphi),'-',color = 'red',label=r'$Re[\nabla \phi]$')
        plt.plot(zgrid,np.imag(gradphi),'-.',color = 'red',label=r'$Im[\nabla \phi]$')
        plt.plot(zgrid,-np.real(omega_complex*apar),'-',color = 'black',label=r'$Re[\omega A_{||}]$')
        plt.plot(zgrid,-np.imag(omega_complex*apar),'-.',color = 'black',label=r'$Im[\omega A_{||}]$')
        plt.xlabel(r'$z/\pi$',size=18)
        plt.legend()
        plt.show()
    
    
    
    
else:  #x_local = False

    dz = 2.0/field.nz
    zgrid = np.arange(field.nz)/float(field.nz-1)*(2.0-dz)-1.0
    zgrid_ext = np.arange(field.nz+4)/float(field.nz+4-1)*(2.0+3*dz)-(1.0+2.0*dz)
    #print zgrid
    #print zgrid_ext
    if 'lx_a' in pars:
        xgrid = np.arange(field.nx)/float(field.nx-1)*pars['lx_a']+pars['x0']-pars['lx_a']/2.0
    else:
        xgrid = np.arange(field.nx)/float(field.nx-1)*pars['lx'] - pars['lx']/2.0
    gpars,geometry = read_geometry_global(pars['magn_geometry'][1:-1]+suffix)

    #Find rational q surfaces
    qmin = np.min(geometry['q'])
    qmax = np.max(geometry['q'])
    mmin = math.ceil(qmin*pars['n0_global'])
    mmax = math.floor(qmax*pars['n0_global'])
    mnums = np.arange(mmin,mmax+1)
    print("mnums",mnums)
    qrats = mnums/float(pars['n0_global'])
    print("qrats",qrats)
    zgridm = np.arange(mmax*20)/float(mmax*20)*2.0-1.0
    nm = int(mmax*20)

    #print "Jacobian",np.shape(geometry['jacobian'])
    #print "Bfield",np.shape(geometry['Bfield'])
    #Note jacobian==> (nz0,nx0)
    phase = (0.0+1.0J)*pars['n0_global']*2.0*np.pi*geometry['q']
    
    phi_bnd = np.zeros((field.nz+4,field.ny,field.nx),dtype = 'complex128')     
    phi_theta = np.zeros((abs(nm),field.ny,field.nx),dtype = 'complex128')     
    phi_m = np.zeros((abs(nm),field.ny,field.nx),dtype = 'complex128')     
    #phi_m0 = np.zeros((nm/2,field.ny,field.nx),dtype = 'complex128')     
    gradphi= np.zeros((field.nz+4,field.ny,field.nx),dtype = 'complex128')     
    phi_bnd[2:-2,:,:] = field.phi()
    for i in range(field.nx):
        phi_bnd[-2,:,i] = phi_bnd[2,:,i]*np.e**(-phase[i])
        phi_bnd[-1,:,i] = phi_bnd[3,:,i]*np.e**(-phase[i])
        phi_bnd[0,:,i] = phi_bnd[-4,:,i]*np.e**(phase[i])
        phi_bnd[1,:,i] = phi_bnd[-3,:,i]*np.e**(phase[i])
        gradphi[:,0,i] = fd_d1_o4(phi_bnd[:,0,i],zgrid_ext)
        gradphi[2:-2:,0,i] = gradphi[2:-2,0,i]/np.pi/(geometry['jacobian'][:,i]*geometry['Bfield'][:,i])

    #field.set_time(field.tfld[-1],len(field.tfld)-1)
    field.set_time(field.tfld[itime])


    imax = np.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(field.nz,field.nx))
    #print "imax",imax

    if plot_ballooning:
        plt.figure(figsize=(8.0,9.5))
        fig=plt.gcf()
        fig.subplots_adjust(right=0.9)
        fig.subplots_adjust(left=0.16)
        fig.subplots_adjust(hspace=0.35)
        plt.subplot(3,1,1)
        plt.ylabel(r'$z/\pi$',fontsize=13)
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.title(r'$|\phi|$')
        plt.contourf(xgrid,zgrid_ext,np.abs(phi_bnd[:,0,:]),70)
        for i in range(len(qrats)):
            ix = np.argmin(abs(geometry['q']-qrats[i])) 
            plt.axvline(xgrid[ix],color='white')
        plt.plot(xgrid[imax[1]],zgrid[imax[0]],'x')
        cb1=plt.colorbar()
        plt.subplot(3,1,2)
        plt.title(r'$Re[\phi]$')
        plt.ylabel(r'$z/\pi$',fontsize=13)
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.contourf(xgrid,zgrid_ext,np.real(phi_bnd[:,0,:]),70)
        plt.colorbar()
        plt.subplot(3,1,3)
        plt.title(r'$Im[\phi]$')
        plt.ylabel(r'$z/\pi$',fontsize=13)
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.contourf(xgrid,zgrid_ext,np.imag(phi_bnd[:,0,:]),70)
        plt.colorbar()
        plt.show()

    if plot_ballooning:
        plt.xlabel(r'$z/\pi$',fontsize=13)
        plt.ylabel(r'$|\phi|$')
        plt.plot(zgrid_ext,np.abs(phi_bnd[:,0,int(pars['nx0']/4)]),label='nx0/4')
        plt.plot(zgrid_ext,np.abs(phi_bnd[:,0,int(pars['nx0']/2)]),label='nx0/2')
        plt.plot(zgrid_ext,np.abs(phi_bnd[:,0,3*int(pars['nx0']/4)]),label='3/4*nx0')
        plt.legend()
        plt.show()

    #Extract m's
    #phi_theta = np.zeros((field.nz+4,field.ny,field.nx),dtype = 'complex128')     
    for i in range(len(xgrid)):
        #print "m",pars['n0_global']*geometry['q'][i]
        #phi_theta[:,0,i] = interp(zgrid_ext,phi_bnd[:,0,i]*np.e**(-1J*pars['n0_global']*geometry['q'][i]*np.pi*zgrid_ext),zgridm)
        phi_theta[:,0,i] = interp(zgrid_ext,phi_bnd[:,0,i],zgridm)
        phi_theta[:,0,i] = phi_theta[:,0,i]*np.e**(1J*pars['n0_global']*geometry['q'][i]*np.pi*zgridm)
        #phi_theta[:,0,i] = interp(zgrid_ext,np.e**(-1J*pars['n0_global']*geometry['q'][i]*np.pi*zgrid_ext),zgridm)
    for i in range(len(xgrid)):
        phi_m[:,0,i] = np.fft.fft(phi_theta[:,0,i])
    #for i in range(len(nm/2)):
    #    phi_m0[i,0,:] = phi_m[i,0,:]+
    if plot_theta:
        for i in range(int(mmin),int(mmax)+1):
            plt.plot(xgrid,np.abs(phi_m[i,0,:]))
        for i in range(len(qrats)):
            ix = np.argmin(abs(geometry['q']-qrats[i])) 
            plt.axvline(xgrid[ix],color='black')
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.ylabel(r'$\phi_{m}$',fontsize=13)
        plt.title(r'$\phi_m$')
        plt.show()

        #plt.figure(figsize=(8.0,9.5))
        #fig=plt.gcf()
        #fig.subplots_adjust(right=0.9)
        #fig.subplots_adjust(left=0.16)
        #fig.subplots_adjust(hspace=0.35)
        #plt.subplot(3,1,1)
        #plt.ylabel(r'$z/\pi$',fontsize=13)
        #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        #plt.title(r'$|\phi(\theta)|$')
        #plt.contourf(xgrid,zgridm,np.abs(phi_theta[:,0,:]),70)
        #for i in range(len(qrats)):
        #    ix = np.argmin(abs(geometry['q']-qrats[i])) 
        #    plt.axvline(xgrid[ix],color='white')
        #    #plt.annotate(
        #plt.plot(xgrid[imax[1]],zgrid[imax[0]],'x')
        #cb1=plt.colorbar()
        #plt.subplot(3,1,2)
        #plt.title(r'$Re[\phi(\theta)]$')
        #plt.ylabel(r'$z/\pi$',fontsize=13)
        #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        #plt.contourf(xgrid,zgridm,np.real(phi_theta[:,0,:]),70)
        #plt.colorbar()
        #plt.subplot(3,1,3)
        #plt.title(r'$Im[\phi(\theta)]$')
        #plt.ylabel(r'$z/\pi$',fontsize=13)
        #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        #plt.contourf(xgrid,zgridm,np.imag(phi_theta[:,0,:]),70)
        #plt.colorbar()
        #plt.show()

    #plt.figure(figsize=(8.0,9.5))
    #mgrid = np.arange(nm)
    #fig=plt.gcf()
    #fig.subplots_adjust(right=0.9)
    #fig.subplots_adjust(left=0.16)
    #fig.subplots_adjust(hspace=0.35)
    #plt.subplot(3,1,1)
    #plt.ylabel('m',fontsize=13)
    #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
    #plt.title(r'$|\phi(\theta)|$')
    #plt.contourf(xgrid,mgrid,np.log(np.abs(phi_m[:,0,:])),70)
    #for i in range(len(qrats)):
    #    ix = np.argmin(abs(geometry['q']-qrats[i])) 
    #    plt.axvline(xgrid[ix],color='white')
    #    #plt.annotate(
    #cb1=plt.colorbar()
    #plt.subplot(3,1,2)
    #plt.title(r'$Re[\phi(\theta)]$')
    ##plt.ylabel(r'$z/\pi$',fontsize=13)
    #plt.ylabel('m',fontsize=13)
    #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
    #plt.contourf(xgrid,mgrid,np.real(phi_m[:,0,:]),70)
    #plt.colorbar()
    #plt.subplot(3,1,3)
    #plt.title(r'$Im[\phi(\theta)]$')
    ##plt.ylabel(r'$z/\pi$',fontsize=13)
    #plt.ylabel('m',fontsize=13)
    #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
    #plt.contourf(xgrid,mgrid,np.imag(phi_m[:,0,:]),70)
    #plt.colorbar()
    #plt.show()


    apar = field.apar()[:,0,:]
    apar_theta = np.zeros((nm,field.nx),dtype = 'complex128')     
    apar_m = np.zeros((nm,field.nx),dtype = 'complex128')     

    #Extract m's
    #phi_theta = np.zeros((field.nz+4,field.ny,field.nx),dtype = 'complex128')     
    for i in range(len(xgrid)):
        #print "m",pars['n0_global']*geometry['q'][i]
        #phi_theta[:,0,i] = interp(zgrid_ext,phi_bnd[:,0,i]*np.e**(-1J*pars['n0_global']*geometry['q'][i]*np.pi*zgrid_ext),zgridm)
        apar_theta[:,i] = interp(zgrid,apar[:,i],zgridm)
        apar_theta[:,i] = apar_theta[:,i]*np.e**(1J*pars['n0_global']*geometry['q'][i]*np.pi*zgridm)
        #phi_theta[:,0,i] = interp(zgrid_ext,np.e**(-1J*pars['n0_global']*geometry['q'][i]*np.pi*zgrid_ext),zgridm)
    for i in range(len(xgrid)):
        apar_m[:,i] = np.fft.fft(apar_theta[:,i])
    #for i in range(len(nm/2)):
    #    phi_m0[i,0,:] = phi_m[i,0,:]+
    if plot_theta:
        for i in range(int(mmin),int(mmax)+1):
            plt.plot(xgrid,np.abs(apar_m[i,:]))
        for i in range(len(qrats)):
            ix = np.argmin(abs(geometry['q']-qrats[i])) 
            plt.axvline(xgrid[ix],color='black')
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.ylabel(r'$A_{||m}$',fontsize=13)
        plt.title(r'$A_{||m}$')
        plt.show()

        #plt.figure(figsize=(4.5,3.0))
        #fig=plt.gcf()
        #plt.subplots_adjust(left=0.16)
        #plt.subplots_adjust(bottom=0.16)
        #plt.ylabel(r'$\theta/\pi$',fontsize=13)
        #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        #plt.title(r'$|A_{||}(\theta)|$')
        #plt.contourf(xgrid,zgridm,np.abs(apar_theta)/1.0e6,70)
        #for i in range(len(qrats)):
        #    ix = np.argmin(abs(geometry['q']-qrats[i])) 
        #    plt.axvline(xgrid[ix],color='white')
        #cb1=plt.colorbar()
        #plt.show()


        #plt.figure(figsize=(8.0,9.5))
        #fig=plt.gcf()
        #fig.subplots_adjust(right=0.9)
        #fig.subplots_adjust(left=0.16)
        #fig.subplots_adjust(hspace=0.35)
        #plt.subplot(3,1,1)
        #plt.ylabel(r'$z/\pi$',fontsize=13)
        #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        #plt.title(r'$|A_{||}(\theta)|$')
        #plt.contourf(xgrid,zgridm,np.abs(apar_theta),70)
        #for i in range(len(qrats)):
        #    ix = np.argmin(abs(geometry['q']-qrats[i])) 
        #    plt.axvline(xgrid[ix],color='white')
        #cb1=plt.colorbar()
        #plt.subplot(3,1,2)
        #plt.title(r'$Re[A_{||}(\theta)]$')
        #plt.ylabel(r'$z/\pi$',fontsize=13)
        #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        #plt.contourf(xgrid,zgridm,np.real(apar_theta),70)
        #plt.colorbar()
        #plt.subplot(3,1,3)
        #plt.title(r'$Im[A_{||}(\theta)]$')
        #plt.ylabel(r'$z/\pi$',fontsize=13)
        #plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        #plt.contourf(xgrid,zgridm,np.imag(apar_theta),70)
        #plt.colorbar()
        #plt.show()




    if plot_ballooning:
        plt.figure(figsize=(8.0,9.5))
        fig=plt.gcf()
        fig.subplots_adjust(right=0.9)
        fig.subplots_adjust(left=0.16)
        fig.subplots_adjust(hspace=0.35)
        plt.subplot(3,1,1)
        plt.ylabel(r'$z/\pi$',fontsize=13)
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.title(r'$|A_{||}|$')
        plt.contourf(xgrid,zgrid,np.abs(apar),70)
        for i in range(len(qrats)):
            ix = np.argmin(abs(geometry['q']-qrats[i])) 
            plt.axvline(xgrid[ix],color='white')
        cb1=plt.colorbar()
        plt.subplot(3,1,2)
        plt.title(r'$Re[A_{||}]$')
        plt.ylabel(r'$z/\pi$',fontsize=13)
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.contourf(xgrid,zgrid,np.real(apar),70)
        plt.colorbar()
        plt.subplot(3,1,3)
        plt.title(r'$Im[A_{||}]$')
        plt.ylabel(r'$z/\pi$',fontsize=13)
        plt.xlabel(r'$\rho_{tor}$',fontsize=13)
        plt.contourf(xgrid,zgrid,np.imag(apar),70)
        plt.colorbar()
        plt.show()

    if calc_epar:

       if os.path.isfile('omega'+suffix):
           om = np.genfromtxt('omega'+suffix)
       else:
           #call(['calc_omega_from_field.py',suffix[1:],'-a'])
           call(['calc_omega_from_field.py',suffix[1:]])
           om = np.genfromtxt('omega'+suffix)

       if om.any():
           if om[0] == 0.0 or om[1] == 0.0 or om[1] != om[1]:
               #call(['calc_omega_from_field.py',suffix[1:],'-a'])
               call(['calc_omega_from_field.py',suffix[1:]])
               om = np.genfromtxt('omega'+suffix)
       else:
           #call(['calc_omega_from_field.py',suffix[1:],'-a'])
           call(['calc_omega_from_field.py',suffix[1:]])
           om = np.genfromtxt('omega'+suffix)

    
       omega_complex = (om[2]*(0.0+1.0J) + om[1])
    
    #if plot_all:
    #    plt.figure(figsize=(8.0,9.5))
    #    fig=plt.gcf()
    #    fig.subplots_adjust(right=0.9)
    #    fig.subplots_adjust(left=0.16)
    #    fig.subplots_adjust(hspace=0.35)
    #    plt.subplot(3,1,1)
    #    plt.plot(zgrid,(np.real(gradphi[2:-2,0,field.nx/4])),color = 'black')
    #    plt.plot(zgrid,(np.real(-omega_complex*field.apar()[:,0,field.nx/4])),color = 'red')
    #    plt.plot(zgrid,(np.imag(gradphi[2:-2,0,field.nx/4])),'-.',color = 'black')
    #    plt.plot(zgrid,(np.imag(-omega_complex*field.apar()[:,0,field.nx/4])),'-.',color = 'red')
    #    plt.subplot(3,1,2)
    #    plt.plot(zgrid,(np.real(gradphi[2:-2,0,field.nx/2])),color = 'black')
    #    plt.plot(zgrid,(np.real(-omega_complex*field.apar()[:,0,field.nx/2])),color = 'red')
    #    plt.plot(zgrid,(np.imag(gradphi[2:-2,0,field.nx/2])),'-.',color = 'black')
    #    plt.plot(zgrid,(np.imag(-omega_complex*field.apar()[:,0,field.nx/2])),'-.',color = 'red')
    #    plt.subplot(3,1,3)
    #    plt.plot(zgrid,(np.real(gradphi[2:-2,0,3*field.nx/4])),color = 'black')
    #    plt.plot(zgrid,(np.real(-omega_complex*field.apar()[:,0,3*field.nx/4])),color = 'red')
    #    plt.plot(zgrid,(np.imag(gradphi[2:-2,0,3*field.nx/4])),'-.',color = 'black')
    #    plt.plot(zgrid,(np.imag(-omega_complex*field.apar()[:,0,3*field.nx/4])),'-.',color = 'red')
    #    plt.show()
    #    plt.contourf(np.abs(gradphi)[:,0,:],50)
    #    plt.title('grad phi')
    #    plt.xlabel('x')
    #    plt.xlabel('z')
    #    plt.colorbar()
    #    plt.show()
    #    plt.contourf(np.abs(omega_complex*field.apar()[:,0,:]),50)
    #    plt.title('omega x Apar')
    #    plt.xlabel('x')
    #    plt.xlabel('z')
    #    plt.colorbar()
    #    plt.show()
    
       print("omega_complex",omega_complex)
       #print "np.shape(field.apar())",np.shape(field.apar())
       if 'ExBrate' in pars and pars['ExBrate'] == -1111: 
           if idb_file == 'empty':
               idb_file = input("Enter ITERDB file name:\n")
           if prof_file == 'empty':
               prof_file = input("Enter gene output profiles file name:\n")
           rhot_idb,profs_idb,units_idb = read_iterdb(idb_file)
           profs = np.genfromtxt(prof_file)
           omegator0 = interp(rhot_idb['VROT'],profs_idb['VROT'],profs[:,0])
           mi = 1.673e-27
           ee = 1.602e-19
           mref = pars['mref']*mi
           time_ref = pars['Lref']/(pars['Tref']*1000.0*ee/mref)**0.5
           apar_cont = 0.0
           diff = 0.0
           apar_cont_2D = np.empty(np.shape(field.apar()),dtype='complex128')
           for i in range(pars['nx0']):
               diff += np.sum(np.abs(gradphi[2:-2,:,i] + (omega_complex+(0.0+1.0J)*pars['n0_global']*omegator0[i]*time_ref)*field.apar()[:,:,i]))
               apar_cont += np.sum(np.abs((omega_complex+(0.0+1.0J)*pars['n0_global']*omegator0[i]*time_ref)*field.apar()[:,:,i]))
               apar_cont_2D[:,:,i] =  (omega_complex+(0.0+1.0J)*pars['n0_global']*omegator0[i]*time_ref)*field.apar()[:,:,i]
       else:
           diff = np.sum(np.abs(gradphi[2:-2,:,:] + omega_complex*field.apar()[:,:,:]))
           apar_cont = np.sum(np.abs(omega_complex*field.apar()[:,:,:]))
       phi_cont = np.sum(np.abs(gradphi[2:-2,:,:]))
       print("diff",diff)
       print("phi_cont",phi_cont)
       print("apar_cont",apar_cont)
       print("diff/abs",diff/(phi_cont+apar_cont))
       #for i in range(pars['nx0']/10-2):
    #    print "i"
    #    for i in range(pars['nx0']/10-2):
    #    plt.plot(zgrid,(np.real(gradphi[2:-2,0,i*field.nx/10])),color = 'black',label='Re gradphi')
    #    #plt.plot(zgrid,(np.real(-omega_complex*field.apar()[:,0,i*field.nx/10])),color = 'red')
    #    plt.plot(zgrid,(np.real(-(omega_complex+(0.0+1.0J)*pars['n0_global']*omegator0[i*field.nx/10]*time_ref)*field.apar()[:,0,i*field.nx/10])),color='red',label='Real omega Apar')
    #    plt.plot(zgrid,(np.imag(gradphi[2:-2,0,i*field.nx/10])),'-.',color = 'black',label='Im gradphi')
    #    #plt.plot(zgrid,(np.imag(-omega_complex*field.apar()[:,0,i*field.nx/10])),'-.',color = 'red')
    #    #plt.plot(zgrid,(np.imag(-(omega_complex+(0.0+1.0J)*pars['n0_global']*omegator0[i*field.nx/10]*time_ref)*field.apar()[:,0,i*field.nx/10])),'-.'color='red')
    #    plt.plot(zgrid,(np.imag(-(omega_complex+(0.0+1.0J)*pars['n0_global']*omegator0[i*field.nx/10]*time_ref)*field.apar()[:,0,i*field.nx/10])),'-.',color='red',label='Im omega Apar')
    #    plt.legend()
    #    plt.show()

       if 'ExBrate' in pars and pars['ExBrate'] == -1111 and plot_ballooning: 
           plt.figure(figsize=(8.0,9.5))
           fig=plt.gcf()
           fig.subplots_adjust(right=0.9)
           fig.subplots_adjust(left=0.16)
           fig.subplots_adjust(hspace=0.35)
           plt.subplot(3,1,1)
           plt.ylabel(r'$z/\pi$',fontsize=13)
           plt.xlabel(r'$\rho_{tor}$',fontsize=13)
           #plt.title(r'$Re(\nabla \phi)$')
           plt.title(r'$Re(grad phi)$')
           plt.contourf(xgrid,zgrid,np.real(gradphi[2:-2,0,:]),70,vmin = np.min(np.real(gradphi[2:-2,0,:])),vmax = np.max(np.real(gradphi[2:-2,0,:])))
           for i in range(len(qrats)):
               ix = np.argmin(abs(geometry['q']-qrats[i])) 
               plt.axvline(xgrid[ix],color='white')
           plt.plot(xgrid[imax[1]],zgrid[imax[0]],'x')
           cb1=plt.colorbar()
           plt.subplot(3,1,2)
           #plt.title(r'$Re[\phi]$')
           plt.title(r'$Re[ omega Apar]$')
           plt.ylabel(r'$z/\pi$',fontsize=13)
           plt.xlabel(r'$\rho_{tor}$',fontsize=13)
           plt.contourf(xgrid,zgrid,-np.real(apar_cont_2D[:,0,:]),70,vmin = np.min(np.real(gradphi[2:-2,0,:])),vmax = np.max(np.real(gradphi[2:-2,0,:])))
           plt.colorbar()
           plt.subplot(3,1,3)
           plt.title(r'$Re[Diff]$')
           plt.ylabel(r'$z/\pi$',fontsize=13)
           plt.xlabel(r'$\rho_{tor}$',fontsize=13)
           plt.contourf(xgrid,zgrid,np.real(gradphi[2:-2,0,:]+apar_cont_2D[:,0,:]),70,vmin = np.min(np.real(gradphi[2:-2,0,:])),vmax = np.max(np.real(gradphi[2:-2,0,:])))
           plt.colorbar()
           plt.show()
    
    
           plt.figure(figsize=(8.0,9.5))
           fig=plt.gcf()
           fig.subplots_adjust(right=0.9)
           fig.subplots_adjust(left=0.16)
           fig.subplots_adjust(hspace=0.35)
           plt.subplot(3,1,1)
           plt.ylabel(r'$z/\pi$',fontsize=13)
           plt.xlabel(r'$\rho_{tor}$',fontsize=13)
           #plt.title(r'$Re(\nabla \phi)$')
           plt.title(r'$Im(grad phi)$')
           plt.contourf(xgrid,zgrid,np.imag(gradphi[2:-2,0,:]),70,vmin = np.min(np.imag(gradphi[2:-2,0,:])),vmax = np.max(np.imag(gradphi[2:-2,0,:])))
           for i in range(len(qrats)):
               ix = np.argmin(abs(geometry['q']-qrats[i])) 
               plt.axvline(xgrid[ix],color='white')
           plt.plot(xgrid[imax[1]],zgrid[imax[0]],'x')
           cb1=plt.colorbar()
           plt.subplot(3,1,2)
           #plt.title(r'$Re[\phi]$')
           plt.title(r'$Im[ omega Apar]$')
           plt.ylabel(r'$z/\pi$',fontsize=13)
           plt.xlabel(r'$\rho_{tor}$',fontsize=13)
           plt.contourf(xgrid,zgrid,-np.imag(apar_cont_2D[:,0,:]),70,vmin = np.min(np.imag(gradphi[2:-2,0,:])),vmax = np.max(np.imag(gradphi[2:-2,0,:])))
           plt.colorbar()
           plt.subplot(3,1,3)
           plt.title(r'$Im[Diff]$')
           plt.ylabel(r'$z/\pi$',fontsize=13)
           plt.xlabel(r'$\rho_{tor}$',fontsize=13)
           plt.contourf(xgrid,zgrid,np.imag(gradphi[2:-2,0,:]+apar_cont_2D[:,0,:]),70,vmin = np.min(np.imag(gradphi[2:-2,0,:])),vmax = np.max(np.imag(gradphi[2:-2,0,:])))
           plt.colorbar()
           plt.show()
    
