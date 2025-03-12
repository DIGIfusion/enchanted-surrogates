import os
import ast
from .base import Parser
import numpy as np
import scipy.stats as stats
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

    def write_input_file(self, params: dict, run_dir: str, base_input_file_path: str):
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
        except:
            print("Give the base case input file as DREAM input hdf5")

        # Initialize the input and output files as run_dir/input.h5 and 
        # run_dir/output.h5.
        if os.path.exists(run_dir):
            input_path = os.path.join(run_dir, 'input.h5')
        else:
            raise FileNotFoundError(f"Couldnt find {run_dir}")

        ds.output.setFilename('output.h5')

        # Anything that the user aims to sample, should be defined
        # here through options.           
        if 'V_loop_wall' in params and 'tau_V_loop' in params:
            tvec = np.linspace(0, ds.timestep.tmax, ds.timestep.nt)
            Vl = params['V_loop_wall']*np.exp(-tvec/params['tau_V_loop'])
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
            sumdb = np.sum(dBr)
            dBr = params['dBB']*dBr/sumdb
            dBr = np.array([[dBr], [dBr]]).reshape(2, 30)
            ds.eqsys.f_re.transport.setMagneticPerturbation(dBr, t=tgrid, r=r)
            ds.eqsys.f_re.transport.setBoundaryCondition(
                DREAM.Settings.Equations.DistributionFunction.BC_F_0,
                )

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
        
        # This computes a distance metric for the current.
        # Mean absolute error is chosen here.
        outputdict = self.read_output_file(run_dir)
        t = np.array(outputdict['output'].eqsys.grid.t[:])
        ip = np.array(outputdict['output'].eqsys.I_p[:])
        iptrace = h5py.File(observations['IP'])
        tobs = iptrace['Ip_time'][:]-48.056
        ipobs = np.abs(iptrace['Ip'])
        func = interp1d(tobs, ipobs)
        ipobs = func(t)
        distance = mean_absolute_error(ip, ipobs)/np.max(ipobs)       
        ipdistance = np.array([distance])
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
            dist=np.max(feature.match_template(image_array[count,:], value, pad_input=True))
            imagedists.append(dist)
        imagedists = np.array(imagedists)
        distances = np.concatenate((ipdistance, imagedists))
        inputnd = []
        for key in inputs:
            inputnd.append(inputs[key])
        inputnd = np.array(inputnd)
        outputdict = {'inputs':inputnd, 'distances':distances}
        # This is a plotting routine that can be used to visually inspect the agreement.
        if plot_comparison:
            fig, axs = plt.subplots(2,3)
            axs[0,0].plot(t, ipobs,'k')
            axs[0,0].plot(t, ip, 'r')
            axs[0,0].set_ylim(0, 1.0e6)
            axs[1,0].axis('off')
            axs[0,1].contourf(images[0])
            axs[0,1].set_xlim(-30,90)
            axs[0,1].set_ylim(-30,90)
            axs[0,1].axis('off')
            axs[1,1].contourf(images[1])
            axs[1,1].set_xlim(-30,90)
            axs[1,1].set_ylim(-30,90)
            axs[1,1].axis('off')
            axs[0,2].contourf(image_array[0,:])
            axs[0,2].set_xlim(0,120)
            axs[0,2].set_ylim(0,120)
            axs[0,2].axis('off')
            axs[1,2].contourf(image_array[1,:])
            axs[1,2].set_xlim(0,120)
            axs[1,2].set_ylim(0,120)
            axs[1,2].axis('off')
            plt.show()
        return outputdict
        

















