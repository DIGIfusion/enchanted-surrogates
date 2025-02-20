import numpy as np

def get_nrg0(suffix,nspec=2,ncols=10,path=''):
    #if nspec < 1 or nspec > 2:
    #    stop

    time=np.empty(0,dtype='float')
    nrg1=np.empty((0,10),dtype='float')

    if nspec<=2:
        nrg2=np.empty((0,10),dtype='float')
    if nspec<=3:
        nrg2=np.empty((0,10),dtype='float')
        nrg3=np.empty((0,10),dtype='float')
    if nspec>=4:
        print( "nspec=",nspec)
        print( "Error, nspec must be less than 4")
        stop
    #qies=np.empty((0,10),dtype='float')
    #qees=np.empty((0,10),dtype='float')
    #qeem=np.empty((0,10),dtype='float')
    f=open(path+'nrg'+suffix,'r')
    nrg_in=f.read()
    nrg0=np.empty((1,ncols))
    #print nrg_in
    nrg_in_lines=nrg_in.split('\n')
    for j in range(len(nrg_in_lines)):
        if nrg_in_lines[j] and j % (nspec+1) == 0:
            time=np.append(time,float(nrg_in_lines[j]))
        elif nrg_in_lines[j] and j % (nspec+1) == 1:
            nline=nrg_in_lines[j].split()
            for i in range(ncols):
                nrg0[0,i]=nline[i]
            nrg1=np.append(nrg1,nrg0,axis=0)
        elif nspec>=2 and nrg_in_lines[j] and j % (nspec+1) ==2:
            nline=nrg_in_lines[j].split()
            for i in range(ncols):
                nrg0[0,i]=nline[i]
            nrg2=np.append(nrg2,nrg0,axis=0)
        elif nspec==3 and nrg_in_lines[j] and j % (nspec+1) ==3:
            nline=nrg_in_lines[j].split()
            for i in range(ncols):
                nrg0[0,i]=nline[i]
            nrg3=np.append(nrg3,nrg0,axis=0)

    if nspec==1:
        return time,nrg1
    elif nspec==2:
        return time,nrg1,nrg2
    else:
        return time,nrg1,nrg2,nrg3

