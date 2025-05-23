import os
import ast
from .base import Parser
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
import scipy.io as sio
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error
from skimage import filters, transform, feature
import matplotlib.pyplot as plt
import yaml
import h5py
import pickle as pkl
try: 
    import DREAM
except ImportError as e:
    print(
        "Cannot find DREAM.",
        "Please add the $DREAMPATH/py to your PYTHONPATH environment variable before running.",
        str(e),
    )
    

class DREAMparser(Parser):
    """
    An I/O parser for DREAM.

    Methods:
        __init__()
            Initializes the DREAMparser object.
        write_input_file(params: dict, run_dir: str) -> None
            Writes a sample input file.
        read_output_file(params: dict, run_dir: str) -> dict
            Reads the output file containing the input parameters.

    """

    def __init__(self):
        """
        Initializes the DREAMparser object.

        """
        pass

    def write_input_file(self, 
                         params: dict, 
                         run_dir: str, 
                         base_input_file_path:str = ' ', 
                         nt:int = 800, 
                         non_lin_solve:bool = True,
                         CQ:bool = False,
                         exp_file_path:str = 'None',
                         re_grid:bool = False,
                         F0:bool = True,
                         ):
        """
        Writes a sample input file.

        Args:
            params (dict): Dictionary containing input parameters.
            run_dir (str): Directory where the input file is written.
            base_input_file_path (str): Path to a basecase input file .h5.

        Returns:
            None

        """

        # Initialize the DREAM settings and load the base input settings.
        ds = DREAM.DREAMSettings()
        try:
            ds.load(base_input_file_path)
            ds.fromOutput(base_input_file_path)
        except:
            print("Give the base case input file as DREAM input hdf5")

        # Initialize the input and output files as run_dir/input.h5 and 
        # run_dir/output.h5.
        if os.path.exists(run_dir):
            input_path = os.path.join(run_dir, 'input.h5')
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")
        if CQ:
            tvec = np.linspace(0, 30.6e-3, nt)
            Tf = []
            t_T_f = params['t_T_f']
            for i in tvec:
                if i < t_T_f:
                    tvalue = params['Tf1'] - (i/t_T_f)*(params['Tf1'] - params['Tf2'])
                else: 
                    tvalue = params['Tf2']
                Tf.append(np.linspace(tvalue, tvalue, 30))
            Tf = np.array(Tf)
            t_Tf = np.array(tvec)
            data = sio.loadmat(exp_file_path)
            dNe = np.squeeze(data['dNe'])
            ne_profile = dNe[:,2]*1e19
            r = np.linspace(0, 63/100, 30)
            r_exp = np.squeeze(dNe[:,0]) - np.squeeze(dNe[0,0])
            nAr = params['nAr_frac']*1e-2*7.9862e18
            inverse_walltime = 1.0/0.005
            current_ok = False
            ds.collisions.collfreq_type = DREAM.Settings.CollisionHandler.COLLFREQ_TYPE_PARTIALLY_SCREENED
            E_init = 20.8104
            ds.eqsys.E_field.setPrescribedData(E_init,radius=r)
            ds.eqsys.n_re.setEceff(DREAM.Settings.Equations.RunawayElectrons.COLLQTY_ECEFF_MODE_FULL)
            ds.eqsys.T_cold.setPrescribedData(Tf[0,0], radius=r, times=t_Tf)
            Tion = Tf[0,0]
            ds.eqsys.n_i.addIon(name='D2', Z=1, T=Tion, iontype=DREAM.Settings.Equations.IonSpecies.IONS_DYNAMIC_FULLY_IONIZED, n=ne_profile, r=r_exp)
            ds.eqsys.n_i.addIon(name='Ar', Z=18, iontype=DREAM.Settings.Equations.IonSpecies.IONS_PRESCRIBED_NEUTRAL, n=nAr) 
            # Hot-tail grid settings
            ds.hottailgrid.setEnabled(False)

            # Runaway grid settings
            ds.runawaygrid.setEnabled(False)

            # Set up radial grid
            ds.radialgrid.setB0(3.0) 
            ds.radialgrid.setNr(30)
            ds.radialgrid.setMinorRadius(63/100) 
            ds.radialgrid.setWallRadius(73/100)        
    
            # Set solver type
            ds.solver.setType(DREAM.Settings.Solver.NONLINEAR)                       # Semi-implicit time stepping
            #ds.solver.setType(Solver.LINEAR_IMPLICIT)
            #ds.solver.setVerbose(True)                               # Print info from Newton iterations
            #ds.solver.preconditioner.enabled = False
            ds.solver.setLinearSolver(3)

            # Include otherquantities to save to output
            ds.other.include('fluid')

            # Set time stepper
            ds.timestep.setTmax(10e-3)
            ds.timestep.setNt(100)

            # Set file name of output file
            init_output_str = os.path.join(run_dir, 'init_out.h5')
            ds.output.setFilename(init_output_str)
            # This section of the code runs the initialization simulation.
            # The current_ok flag is used to iterate the initial electric field
            # to get the correct current.
            Ip_init=1.421e6
            while current_ok != True:
                if os.path.isfile(init_output_str):
                    os.remove(init_output_str)
                f = DREAM.runiface(ds, outfile=init_output_str)
                current = f['eqsys']['I_p'][-1][0]
                nre = f['eqsys']['j_tot'][:]/np.max(f['eqsys']['j_tot'][:])
                if abs(current - Ip_init) > 1e4:
                    E_init = E_init*(Ip_init/current)
                    ds.eqsys.E_field.setPrescribedData(E_init,radius=r)
                else:
                    current_ok = True
    
            nre = nre[-1]
            n_re = nre*np.exp(params['Nre'])
            ds1 = DREAM.DREAMSettings(ds)
            ds1.fromOutput(os.path.join(run_dir,'init_out.h5'), ignore=['n_re', 'T_cold'])
            ds1.eqsys.T_cold.setPrescribedData(Tf, radius=r, times=t_Tf)
            ds1.eqsys.E_field.setType(DREAM.Settings.Equations.ElectricField.TYPE_SELFCONSISTENT)
            ds1.eqsys.E_field.setBoundaryCondition(bctype = DREAM.Settings.Equations.ElectricField.BC_TYPE_SELFCONSISTENT, 
                                           inverse_wall_time = inverse_walltime,
                                           R0=2.643)
            ds1.eqsys.n_i.getIon('Ar').initialize_dynamic_neutral(
                    interpr=ds.eqsys.n_i.getIon('Ar').r, n=nAr)
            ds1.timestep.setTmax(30.6e-3)
            ds1.timestep.setNt(nt)
            ds1.timestep.setNumberOfSaveSteps(100)
            ds1.eqsys.n_re.setInitialProfile(density=n_re,radius=r)
            ds1.eqsys.n_re.setAvalanche(DREAM.Settings.Equations.RunawayElectrons.AVALANCHE_MODE_FLUID)
            ds1.eqsys.n_re.setDreicer(DREAM.Settings.Equations.RunawayElectrons.DREICER_RATE_DISABLED)
            ds1.hottailgrid.setEnabled(False)
            ds1.runawaygrid.setEnabled(False)
            if re_grid:
                ds1.solver.tolerance.set('j_re', abstol=1)
                ds1.runawaygrid.setEnabled(True)
                ds1.runawaygrid.setNxi(20)
                ds1.runawaygrid.setNp(30)
                ds1.runawaygrid.setPmax(160)
                ds1.runawaygrid.setBiuniformGrid(psep=10, npsep_frac=0.5, thetasep=0.6,nthetasep_frac=0.5)
                f = np.zeros((1,1,1))                                 #nr x np x nXi
                ds1.eqsys.f_re.setInitialValue(f,r=[0],p =[0],xi=[0])
                ds1.eqsys.f_re.setInitType(1) #Fix
                # Collision mode
                ds1.collisions.collfreq_mode = DREAM.Settings.CollisionHandler.COLLFREQ_MODE_ULTRA_RELATIVISTIC
    
                # Flux limiter
                ds1.eqsys.f_re.setAdvectionInterpolationMethod(ad_int=DREAM.Settings.Equations.DistributionFunction.AD_INTERP_TCDF,
                                                               ad_jac=DREAM.Settings.Equations.DistributionFunction.AD_INTERP_JACOBIAN_UPWIND)
                ds1.eqsys.f_re.setSynchrotronMode(DREAM.Settings.Equations.DistributionFunction.SYNCHROTRON_MODE_INCLUDE)
            ds = ds1
        else:
            ds.timestep.setTmax(24.0e-3)
            ds.timestep.setNt(nt)
            ds.timestep.setNumberOfSaveSteps(47)
            # Relatively low discretization to speed up the calculation
            ds.runawaygrid.setNxi(20)
            ds.runawaygrid.setNp(30)
        if non_lin_solve:
            # Use the NONLINEAR solver for robustness
            ds.solver.setType(DREAM.Settings.Solver.NONLINEAR)
        else:
            ds.solver.setType(DREAM.Settings.Solver.LINEAR_IMPLICIT)
        # Anything that the user aims to sample, should be defined
        # here through options.           
        if 'V_loop_wall' in params: 
            tvec = np.linspace(0, ds.timestep.tmax, ds.timestep.nt)
            if 'tau_V_loop' in params:
                Vl = params['V_loop_wall']*np.exp(-tvec/params['tau_V_loop'])
            else:
                Vl = params['V_loop_wall']*np.ones(len(tvec))
            inverse_wall_time = ds.eqsys.E_field.inverse_wall_time
            R = ds.eqsys.E_field.R0
            ds.eqsys.E_field.setBoundaryCondition(
                bctype = DREAM.Settings.Equations.ElectricField.BC_TYPE_TRANSFORMER,
                inverse_wall_time = inverse_wall_time,
                R0=R,
                V_loop_wall_R0 = Vl,
                times = tvec,
                )
        if 'dBB' in params and 'alpha' in params and 'beta' in params:
            tgrid = np.array([0, ds.timestep.tmax])
            r = np.linspace(0, ds.radialgrid.a/100, int(ds.radialgrid.nr))
            dBr = stats.gamma.pdf(r, a=params['alpha'], scale=1.0/params['beta'])
            maxdb = np.max(dBr)
            dBr = 10**(params['dBB'])*dBr/maxdb
            dBr = np.array([[dBr], [dBr]]).reshape(2, 30)
            ds.eqsys.f_re.transport.setMagneticPerturbation(dBr, t=tgrid, r=r)
            if F0:
                ds.eqsys.f_re.transport.setBoundaryCondition(
                    DREAM.Settings.TransportSettings.BC_F_0,
                    )
            else:
                ds.eqsys.f_re.transport.setBoundaryCondition(
                    DREAM.Settings.TransportSettings.BC_DF_CONST,
                    )
        if 'dBB' in params and 'decay' in params:
            tgrid = np.array([0, ds.timestep.tmax])
            r = np.linspace(0, ds.radialgrid.a/100, int(ds.radialgrid.nr))
            dBr = np.exp(r/params['decay']) - 1.0
            maxdb = np.max(dBr)
            dBr = 10**(params['dBB'])*dBr/maxdb
            dBr = np.array([[dBr], [dBr]]).reshape(2, 30)
            ds.eqsys.f_re.transport.setMagneticPerturbation(dBr, t=tgrid, r=r)
            ds.eqsys.f_re.transport.setBoundaryCondition(
                DREAM.Settings.TransportSettings.BC_F_0,
                )
        
        if 'dBB' in params and 'radius_frac' in params:
            tgrid = np.array([0, ds.timestep.tmax])
            r = np.linspace(0, ds.radialgrid.a/100, int(ds.radialgrid.nr))
            dBr = np.zeros(len(r))
            idx = int(params['radius_frac']*len(r))
            dBr[idx:] = 10**(params['dBB'])
            dBr = np.array([[dBr], [dBr]]).reshape(2, 30)
            ds.eqsys.f_re.transport.setMagneticPerturbation(dBr, t=tgrid, r=r)
            ds.eqsys.f_re.transport.setBoundaryCondition(
                DREAM.Settings.TransportSettings.BC_F_0,
                )           
        output_path = os.path.join(run_dir,'output.h5')
        ds.output.setFilename(output_path)
        # Save the input file in run_dir.
        ds.save(input_path)
        print(f"DREAM input written to: {input_path}.")
        with open(os.path.join(run_dir,'params.pkl'),'wb') as paramsdump:
            pkl.dump(params, paramsdump)
            print(f"The applied input vector written in {run_dir}/params.pkl")
          
    def read_output_file(self, run_dir: str):
        """
        Reads the input and output files from the run directory

        Args:
            run_dir (str): Directory where the output file is located.

        Returns:
            dict: Dictionary containing the settings and the output dictionaries.

        Raises:
            FileNotFoundError: If the output file does not exist.

        """
        file_name_input = os.path.join(run_dir, "input.h5")
        file_name_output = os.path.join(run_dir, "output.h5")
        if not os.path.exists(file_name_input):
            raise FileNotFoundError(f"{file_name_input}")
        if not os.path.exists(file_name_output):
            raise FileNotFoundError(f"{file_name_output}")
        ds = DREAM.DREAMSettings()
        do = DREAM.DREAMOutput()
        ds.load(file_name_input)
        do.load(file_name_output)
        outputdict = {'input':ds, 'output':do}
        return outputdict

    def collect_sample_information(self, run_dir: str, observations: dict, plot_comparison=False):
        """
        Reads the information from the run directory that is needed for the 
        Bayesian optimization sampler.

        Args:
            run_dir (str): Directory where the run has been conducted.
            observations (dict): Dictionary of paths to observations to which the output is compared.

        Kwargs:
            plot_comparison: This flag can be set to True to create comparison plots for the discrepancies.

        Returns:
            dict: Dictionary containing the information needed for the BO routines.
        """
        # Read the sample input vector as dumped in the pkl file
        with open(os.path.join(run_dir, 'params.pkl'), 'rb') as paramsfile:
            inputs = pkl.load(paramsfile)
        
        try:
            # This computes a distance metric for the current.
            # Mean absolute error is chosen here.
            outputdict = self.read_output_file(run_dir)
            t = np.array(outputdict['output'].eqsys.grid.t[:])
            t[20]
            ip = np.array(outputdict['output'].eqsys.I_p[:])
            iptrace = h5py.File(observations['IP'])
            if os.path.isfile(os.path.join(run_dir, 'init_out.h5')):
                CQ = True
            else:
                CQ = False
            if CQ:
                tobs = iptrace['IP_data'][0,:]
                ipobs = np.abs(iptrace['IP_data'][1,:])
            else:
                tobs = iptrace['Ip_time'][:]-48.056
                ipobs = np.abs(iptrace['Ip'])
            func = interp1d(tobs, ipobs, fill_value='extrapolate')
            ipobs = func(t)
            distance = mean_absolute_error(ip, ipobs)/np.max(ipobs)       
            ipdistance = np.array([distance])
            if CQ == False:
                # This computes the distance metric for the synchrotron images.
                # This implementation goes digging into the SOFT run directory, breaking the compartmentalization 
                # a little. Future modications may change this to respect the interfaces more elegantly.
                # However, it is all still contained within the DREAM run directory, as SOFT is essentially a 
                # post processor.
                images = []
                for i in observations['images']:
                    image1 = np.load(i)
                    im1 = np.rot90(image1['data'][0].T)
                    # Crop the image. Not very elegant to be honest. However, the experimental image has 
                    # some optical features that are difficult to reproduce synthetically and 
                    # the most straight forward solution is to crop the image above these features.
                    im1 = im1[85:130, 30:110] 
                    images.append(im1)
                images = np.array(images)
                dirlist = os.listdir(run_dir)
                time_idx_list = []
                image_list = []
                for i in dirlist:
                    if ('input' not in i) and ('output' not in i) and ('params' not in i):
                        finp = open(os.path.join(run_dir, i,'soft_input'),'r')
                        text = 'g'
                        while ' time ' not in text:
                            text = finp.readline()
                        t1 = text.split()
                        time_idx = int(t1[2][:-1])
                        soft_ima = h5py.File(os.path.join(run_dir, i,'image_out.h5'))
                        soft_ima = np.rot90(soft_ima['image'])
                        soft_ima = transform.resize(soft_ima, (120,120))
                        time_idx_list.append(time_idx)
                        image_list.append(soft_ima)
                time_idx_array = np.array(time_idx_list)
                image_array = np.array(image_list)
                idxorder = np.argsort(time_idx_array)
                image_array = image_array[idxorder,:]
                imagedists = []
                for count, value in enumerate(images):
                    dist=1.0 - np.max(feature.match_template(image_array[count,:], value, pad_input=True))
                    imagedists.append(dist)
                imagedists = np.array(imagedists)
                distances = np.concatenate((np.log(ipdistance), np.log(imagedists)))
            else:
                distances = ipdistance
            inputnd = []
            for key in inputs:
                inputnd.append(inputs[key])
            inputnd = np.array(inputnd)
            outputdict = {'run_dir':run_dir, 'inputs':inputnd, 'distances':distances, 'failure':0}
        except:
            distances = np.array([-1])
            inputnd = []
            for key in inputs:
                inputnd.append(inputs[key])
            inputnd = np.array(inputnd)
            outputdict = {'run_dir':run_dir, 'inputs':inputnd, 'distances':distances, 'failure':1}
        # This is a plotting routine that can be used to visually inspect the agreement.
        if plot_comparison:
            if CQ == False:
                gm = [(0, 0, 0), (.15, .15, .5), (.3, .15, .75),
                  (.6, .2, .50), (1, .25, .15), (.9, .5, 0),
                  (.9, .75, .1), (.9, .9, .5), (1, 1, 1)]
                colormap = LinearSegmentedColormap.from_list('GM', gm)
                fig, axs = plt.subplots(1,3)
                axs[0].plot(t, ipobs,'k')
                axs[0].plot(t, ip, 'r')
                axs[0].set_ylim(0, 1.0e6)
                axs[0].set_xlim(0, 24.0e-3)
                axs[0].set_aspect(24.0e-3/1.0e6)
                axs[0].set_xlabel('Time (s)')
                axs[0].set_ylabel('Current (A)')
                #axs[1,0].axis('off')
                axs[1].contourf(images[0], cmap=colormap)
                axs[1].set_xlim(-30,90)
                axs[1].set_ylim(-30,90)
                axs[1].axis('off')
                axs[1].set_aspect(1.0)
                #axs[1,1].contourf(images[1])
                #axs[1,1].set_xlim(-30,90)
                #axs[1,1].set_ylim(-30,90)
                #axs[1,1].axis('off')
                axs[2].contourf(image_array[0,:], cmap=colormap)
                axs[2].set_xlim(0,120)
                axs[2].set_ylim(0,120)
                axs[2].axis('off')
                axs[2].set_aspect(1.0)
                #axs[1,2].contourf(image_array[1,:])
                #axs[1,2].set_xlim(0,120)
                #axs[1,2].set_ylim(0,120)
                #axs[1,2].axis('off')
                plt.show()
            else:
                plt.plot(t, ipobs,'k')
                plt.plot(t, ip, 'r')
                plt.ylim(0, 1.5e6)
                plt.xlim(0, 30.6e-3)
                plt.gca().set_aspect(30.6e-3/1.5e6)
                plt.xlabel('Time (s)')
                plt.ylabel('Current (A)')
                plt.show()
        return outputdict


















