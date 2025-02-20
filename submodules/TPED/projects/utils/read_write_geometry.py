import numpy as np
import matplotlib.pyplot as plt
# from scipy import linalg
# import sys

def read_geometry_local(file_name):
    f = open(file_name,'r')
    file_raw = f.read()
    file_lines = file_raw.split('\n')

    parameters = {}
    l = 1
    while '/' not in file_lines[l] and len(file_lines[l])>0:
        lsplit = file_lines[l].split('=')
    #    print lsplit[0].strip()
        if lsplit[0].strip() == 'gridpoints':
            parameters[lsplit[0].strip()] = int(float(lsplit[1].strip()))
        elif lsplit[0].strip() == 'magn_geometry':
            parameters[lsplit[0].strip()] = lsplit[1].strip()[1:-1]
        elif len(lsplit[0]) > 0:
            parameters[lsplit[0].strip()] = float(lsplit[1])
        l += 1
        #print "lsplit",lsplit
    
    #print parameters

    geometry = {}
    #1. ggxx(pi1,pj1,k) 
    geometry['ggxx'] = np.empty(0)
    #2. ggxy(pi1,pj1,k)
    geometry['ggxy'] = np.empty(0)
    #3. ggxz(pi1,pj1,k)
    geometry['ggxz'] = np.empty(0)
    #4. ggyy(pi1,pj1,k) 
    geometry['ggyy'] = np.empty(0)
    #5. ggyz(pi1,pj1,k)
    geometry['ggyz'] = np.empty(0)
    #6. ggzz(pi1,pj1,k)
    geometry['ggzz'] = np.empty(0)
    #7. gBfield(pi1,pj1,k) 
    geometry['gBfield'] = np.empty(0)
    #8. gdBdx(pi1,pj1,k)
    geometry['gdBdx'] = np.empty(0)
    #9. gdBdy(pi1,pj1,k)
    geometry['gdBdy'] = np.empty(0)
    #10. gdBdz(pi1,pj1,k)
    geometry['gdBdz'] = np.empty(0)
    #11. gjacobian(pi1,pj1,k)
    geometry['gjacobian'] = np.empty(0)
    #12. gl_R(pi1,k)
    geometry['gl_R'] = np.empty(0)
    #13. gl_phi(pi1,k)
    geometry['gl_phi'] = np.empty(0)
    #14. gl_z(pi1,k)
    geometry['gl_z'] = np.empty(0)
    #15. gl_dxdR(pi1,k)
    geometry['gl_dxdR'] = np.empty(0)
    #16. gl_dxdZ(pi1,k)
    geometry['gl_dxdZ'] = np.empty(0)

    if 'sign_Ip_CW' in file_raw: 
        l += 4
    else:
        l += 1
    while file_lines[l]:
        line = file_lines[l].split()
        geometry['ggxx'] = np.append(geometry['ggxx'],float(line[0].strip()))
        geometry['ggxy'] = np.append(geometry['ggxy'],float(line[1].strip()))
        geometry['ggxz'] = np.append(geometry['ggxz'],float(line[2].strip()))
        geometry['ggyy'] = np.append(geometry['ggyy'],float(line[3].strip()))
        geometry['ggyz'] = np.append(geometry['ggyz'],float(line[4].strip()))
        geometry['ggzz'] = np.append(geometry['ggzz'],float(line[5].strip()))
        geometry['gBfield'] = np.append(geometry['gBfield'],float(line[6].strip()))
        geometry['gdBdx'] = np.append(geometry['gdBdx'],float(line[7].strip()))
        geometry['gdBdy'] = np.append(geometry['gdBdy'],float(line[8].strip()))
        geometry['gdBdz'] = np.append(geometry['gdBdz'],float(line[9].strip()))
        geometry['gjacobian'] = np.append(geometry['gjacobian'],float(line[10].strip()))
        geometry['gl_R'] = np.append(geometry['gl_R'],float(line[11].strip()))
        geometry['gl_phi'] = np.append(geometry['gl_phi'],float(line[12].strip()))
        geometry['gl_z'] = np.append(geometry['gl_z'],float(line[13].strip()))
        geometry['gl_dxdR'] = np.append(geometry['gl_dxdR'],float(line[14].strip()))
        geometry['gl_dxdZ'] = np.append(geometry['gl_dxdZ'],float(line[15].strip()))
        #print "l",l,float(line[15])
        l += 1
        
    
    #for i in geometry:
    #    plt.title(i)
    #    plt.plot(geometry[i])
    #    plt.show()    

    return parameters, geometry

def read_geometry_global(file_name):
    f = open(file_name,'r')
    file_raw = f.read()
    file_lines = file_raw.split('\n')

    parameters = {}
    #for i in range(11):
    #    lsplit = file_lines[i+1].split('=')
    #    if 'magn' in file_lines[i+1]:
    #        parameters[lsplit[0].strip()] = lsplit[1].strip()[1:-1]
    #    else:
    #        parameters[lsplit[0].strip()] = float(lsplit[1].strip())
    #    print parameters[lsplit[0].strip()]

    l=1
    while '/' not in file_lines[l] and len(file_lines[l])>0:
        lsplit = file_lines[l].split('=')
        #print lsplit[0],lsplit[1]
        if lsplit[0].strip() == 'gridpoints':
            parameters[lsplit[0].strip()] = int(float(lsplit[1].strip()))
        elif lsplit[0].strip() == 'magn_geometry':
            parameters[lsplit[0].strip()] = lsplit[1].strip()[1:-1]
        else:
            parameters[lsplit[0].strip()] = float(lsplit[1])
        l += 1
        print(parameters[lsplit[0].strip()])


    #lsplit = file_lines[11].split('=')
    #parameters[lsplit[0].strip()] = lsplit[1].strip()[1:-1]
    
    print(parameters)
    
    geometry = {}
    geometry['q'] = np.empty(0)
    geometry['gxx'] = np.empty(0)
    geometry['gxy'] = np.empty(0)
    geometry['gxz'] = np.empty(0)
    geometry['gyy'] = np.empty(0)
    geometry['gyz'] = np.empty(0)
    geometry['gzz'] = np.empty(0)
    geometry['Bfield'] = np.empty(0)
    geometry['dBdx'] = np.empty(0)
    geometry['dBdy'] = np.empty(0)
    geometry['dBdz'] = np.empty(0)
    geometry['jacobian'] = np.empty(0)
    geometry['C_y'] = np.empty(0)
    geometry['C_xy'] = np.empty(0)
    geometry['geo_R'] = np.empty(0)
    geometry['geo_Z'] = np.empty(0)
    geometry['geo_c1'] = np.empty(0)
    geometry['geo_c2'] = np.empty(0)
    geometry['dpdx_pm_arr'] = np.empty(0)

    for ln in range(len(file_lines)):
        if file_lines[ln].strip() in geometry:
            this_var = file_lines[ln].strip()
            print(this_var)
            ln2 = ln+1
            this_line = file_lines[ln2]
            while not this_line.strip() in geometry:
                lsplit = this_line.split()
                for i in lsplit:
                    geometry[this_var] = np.append(geometry[this_var],float(i)) 
                ln2 += 1
                if ln2 != len(file_lines):
                    this_line = file_lines[ln2]
                else:
                    #Need to trigger exit
                    this_line = 'q'

    nx0 = int(len(geometry['q']))
    nz0 = int(parameters['gridpoints'])
    #plt.plot(geometry['gxx'])
    #plt.show()
    geometry['gxx']=geometry['gxx'].reshape((nz0,nx0))
    geometry['gxy']=geometry['gxy'].reshape((nz0,nx0))
    geometry['gxz']=geometry['gxz'].reshape((nz0,nx0))
    geometry['gyy']=geometry['gyy'].reshape((nz0,nx0))
    geometry['gyz']=geometry['gyz'].reshape((nz0,nx0))
    geometry['gzz']=geometry['gzz'].reshape((nz0,nx0))
    geometry['Bfield']=geometry['Bfield'].reshape((nz0,nx0))
    geometry['dBdx']=geometry['dBdx'].reshape((nz0,nx0))
    geometry['dBdy']=geometry['dBdy'].reshape((nz0,nx0))
    geometry['dBdz']=geometry['dBdz'].reshape((nz0,nx0))
    geometry['jacobian']=geometry['jacobian'].reshape((nz0,nx0))
    geometry['geo_R']=geometry['geo_R'].reshape((nz0,nx0))
    geometry['geo_Z']=geometry['geo_Z'].reshape((nz0,nx0))
    geometry['geo_c1']=geometry['geo_c1'].reshape((nz0,nx0))
    geometry['geo_c2']=geometry['geo_c2'].reshape((nz0,nx0))

    return parameters, geometry

def write_tracer_efit_file(parameters,geometry,file_name):
    f = open(file_name,'w')
    f.write('&parameters\n')
    f.write('gridpoints =    '+str(int(float(parameters['gridpoints'])))+'\n')
    f.write('q0 =    '+str(parameters['q0'])+'\n')
    f.write('shat =    '+str(parameters['shat'])+'\n')
    f.write('s0 =    '+str(parameters['s0'])+'\n')
    f.write('minor_r =    '+str(parameters['minor_r'])+'\n')
    f.write('major_R =    '+str(parameters['major_R'])+'\n')
    f.write('trpeps =    '+str(parameters['trpeps'])+'\n')
    f.write('beta =    '+str(parameters['beta'])+'\n')
    f.write('Lref =    '+str(parameters['Lref'])+'\n')
    f.write('Bref =    '+str(parameters['Bref'])+'\n')
    f.write('magn_geometry =    '+'\''+str(parameters['magn_geometry'])+'\'\n/\n')

    nz0 = int(float(parameters['gridpoints']))
    gsize = np.shape(geometry['gxx'])    
    nx0 = int(gsize[0]*gsize[1]/nz0)

    #geometry['q'] = np.empty(0)
    f.write('q\n')
    for i in range(nx0):
        f.write("%20.12E"% geometry['q'][i])
        if (i+1) % 3 == 0:
            f.write('\n')

    if len(geometry['q']) % 3 != 0:
        f.write('\n')

    #geometry['gxx'] = np.empty(0)
    temp = np.reshape(geometry['gxx'],nx0*nz0)
    f.write('gxx\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    #geometry['gxy'] = np.empty(0)
    temp = np.reshape(geometry['gxy'],nx0*nz0)
    f.write('gxy\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['gxz'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['gxz'],nx0*nz0)
    f.write('gxz\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['gyy'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['gyy'],nx0*nz0)
    f.write('gyy\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    #geometry['gyz'] = np.empty(0)
    temp = np.reshape(geometry['gyz'],nx0*nz0)
    f.write('gyz\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['gzz'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['gzz'],nx0*nz0)
    f.write('gzz\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['Bfield'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['Bfield'],nx0*nz0)
    f.write('Bfield\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['dBdx'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['dBdx'],nx0*nz0)
    f.write('dBdx\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['dBdy'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['dBdy'],nx0*nz0)
    f.write('dBdy\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['dBdz'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['dBdz'],nx0*nz0)
    f.write('dBdz\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['jacobian'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['jacobian'],nx0*nz0)
    f.write('jacobian\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['C_y'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    f.write('C_y\n')
    for i in range(len(geometry['C_y'])):
        f.write("%20.12E"% geometry['C_y'][i])
        if (i+1) % 3 == 0:
            f.write('\n')
    #geometry['C_xy'] = np.empty(0)

    if len(geometry['C_y']) % 3 != 0:
        f.write('\n')

    f.write('C_xy\n')
    for i in range(len(geometry['C_xy'])):
        f.write("%20.12E"% geometry['C_xy'][i])
        if (i+1) % 3 == 0:
            f.write('\n')

    if len(geometry['C_xy']) % 3 != 0:
        f.write('\n')

    #geometry['geo_R'] = np.empty(0)
    temp = np.reshape(geometry['geo_R'],nx0*nz0)
    f.write('geo_R\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['geo_Z'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['geo_Z'],nx0*nz0)
    f.write('geo_Z\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['geo_c1'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['geo_c1'],nx0*nz0)
    f.write('geo_c1\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['geo_c2'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    temp = np.reshape(geometry['geo_c2'],nx0*nz0)
    f.write('geo_c2\n')
    for i in range(len(temp)):
        f.write("%20.12E"% temp[i])
        if (i+1) % 16 == 0:
            f.write('\n')
    #geometry['dpdx_pm_arr'] = np.empty(0)
    if nx0*nz0 % 16 != 0:
        f.write('\n')
    f.write('dpdx_pm_arr\n')
    for i in range(len(geometry['dpdx_pm_arr'])):
        f.write("%20.12E"% geometry['dpdx_pm_arr'][i])
        if (i+1) % 3 == 0:
            f.write('\n')
    f.close()


