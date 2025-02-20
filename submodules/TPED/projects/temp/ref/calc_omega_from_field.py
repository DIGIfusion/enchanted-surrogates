#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import optparse as op
import matplotlib.pyplot as plt
from fieldlib import *
from get_nrg import *
from ParIO import * 
from finite_differences import *


parser=op.OptionParser(description='Calculates growth rate and frequncy from field file.')
parser.add_option('--apar','-a',action='store_const',const=1,help = 'Calculate from Apar instead of phi.')
options,args=parser.parse_args()
if len(args)!=1:
    exit("""
Please include run number as argument (e.g., 0001)."
    \n""")
calc_from_apar=options.apar
suffix = args[0]
suffix = '_'+suffix

par = Parameters()
par.Read_Pars('parameters'+suffix)
pars = par.pardict
if pars['n_spec'] == 1:
    time, nrgi = get_nrg0(suffix,nspec=1)
elif pars['n_spec'] == 2:
    time, nrgi, nrge = get_nrg0(suffix,nspec=2)
elif pars['n_spec'] == 3:
    time, nrgi, nrge, nrg2 = get_nrg0(suffix,nspec=3)
else:
    sys.exit("n_spec must be 1,2,3.")

print("len(time)",len(time))
print("len(nrgi[:,0])",len(nrgi[:,0]))

if calc_from_apar:
   print( "Calculating growth rate from apar.")
   plt.semilogy(time,nrge[:,7])
   plt.xlabel('time')
   plt.show()
else:
   print( "Calculating growth rate from phi.")
   plt.semilogy(time,nrgi[:,6])
   plt.title('QiES')
   plt.xlabel('time')
   plt.show()

tstart = float(input("Enter start time: "))
tend = float(input("Enter end time: "))

field = fieldfile('field'+suffix,pars)
istart = np.argmin(abs(np.array(field.tfld)-tstart))
print( "istart,start_time",istart,field.tfld[istart])
iend = np.argmin(abs(np.array(field.tfld)-tend))
print( "iend,end_time",iend,field.tfld[iend])


#field.set_time(field.tfld[-1],len(field.tfld)-1)
field.set_time(field.tfld[-1])
imax = np.unravel_index(np.argmax(abs(field.phi()[:,0,:])),(field.nz,field.nx))
phi = np.empty(0,dtype='complex128')
if pars['n_fields'] > 1:
    imaxa = np.unravel_index(np.argmax(abs(field.apar()[:,0,:])),(field.nz,field.nx))
    apar = np.empty(0,dtype='complex128')

print( "imax",imax)

time = np.empty(0)
for i in range(istart,iend):
    #field.set_time(field.tfld[i],i)
    field.set_time(field.tfld[i])
    phi = np.append(phi,field.phi()[imax[0],0,imax[1]])
    if pars['n_fields'] > 1:
        apar = np.append(apar,field.apar()[imaxa[0],0,imaxa[1]])
    time = np.append(time,field.tfld[i])
print( "phi_0,phi_f",phi[0],phi[-1]     )
#plt.semilogy(time,np.abs(phi))
#plt.semilogy(time,np.abs(apar))
#plt.show()
#omega = fd_d1_o4(np.log(phi),time)
if len(phi) < 2.0:
    output_zeros = True
    omega = 0.0+0.0J
else:
    output_zeros = False
    if calc_from_apar:
        print( "Calculating omega from apar")
        if pars['n_fields'] < 2:
            stop
        omega = np.log(apar/np.roll(apar,1))
        dt = time - np.roll(time,1)
        omega /= dt
        omega = np.delete(omega,0)
        time = np.delete(time,0)
    else:
        omega = np.log(phi/np.roll(phi,1))
        dt = time - np.roll(time,1)
        omega /= dt
        print( 'omega',omega)
        omega = np.delete(omega,0)
        time = np.delete(time,0)

gam_avg = np.average(np.real(omega))
om_avg = np.average(np.imag(omega))
print( "Gamma:",gam_avg)
print( "Omega:",om_avg)


if output_zeros:
    f=open('omega'+suffix,'w')
    f.write(str(pars['kymin'])+'    '+str(0.0)+'    '+str(0.0)+'\n')
    f.close()
else:
    plt.plot(time,np.real(omega),label='gamma')
    plt.plot(time,np.imag(omega),label='omega')
    plt.xlabel('t(a/cs)')
    plt.ylabel('omega(cs/a)')
    plt.legend(loc='upper left')
    plt.show()

    selection = int(float(input('How to proceed:\n1. Accept calculation \n2. Manually enter gamma and omega \n3. Don\'t output anything\n')))
    if selection == 1:
        f=open('omega'+suffix,'w')
        f.write(str(pars['kymin'])+'    '+str(gam_avg)+'    '+str(om_avg)+'\n')
        f.close()
    elif selection == 2:
        gam_avg = float(input('Enter gamma: '))
        om_avg = float(input('Enter omega: '))
        f=open('omega'+suffix,'w')
        f.write(str(pars['kymin'])+'    '+str(gam_avg)+'    '+str(om_avg)+'\n')
        f.close()
    
