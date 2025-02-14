import os
import ast
from .base import Parser
import numpy as np
import scipy.stats as stats
import h5py
    
class SOFTparser(Parser):
    """
    An I/O parser for SOFT post processor for DREAM.

    Methods:
        __init__()
            Initializes the SOFTparser object.
        write_input_file(params: dict, run_dir: str) -> None
            Writes a sample input file.
        read_output_file(params: dict, run_dir: str) -> dict
            Reads the output file containing the input parameters.

    """

    def __init__(self):
        """
        Initializes the SOFTparser object.

        """
        pass

    def write_input_file(self, params: dict, run_dir: str):
        """
        Writes a SOFT input file.

        Args:
            params (dict): Dictionary containing input parameters.
            run_dir (str): Directory where the input file is written.

        Returns:
            None

        """
        input_path = os.path.join(run_dir, "soft_input")
        f = open(input_path,'w')
        f.write('\n')
        f.write('magnetic_field     = numeric-field;'+'\n'+
            'tools              = rad;'+'\n'+
            'include_drifts     = True;'+'\n'+
            'distribution_function = distFunc;'+'\n')
        f.write('@MagneticField numeric-field (numeric) {'+'\n'+
            '    filename = "'+params['mag_field_path']+'";'+'\n'+
            '}'+'\n')
        f.write('@ParticleGenerator PGen {'+'\n'+
            '    a  = 0.0, 1, 100; # (m)'+'\n'+
            '    p  = 10, 100, 50; # p is normalized to particle rest mass'+'\n'+
            '    ithetap = 0.02, 0.35, 40;'+'\n'+
            '    progress = 100;'+'\n'+
            '}'+'\n')
        f.write('@ParticlePusher PPusher {'+'\n'+
            '    nt = 2000;        # Number of timesteps per orbit (resolution parameter)'+'\n'+
            '    force_numerical_jacobian = yes;' +'\n'+
            '}'+'\n')
        f.write('@DistributionFunction distFunc(dream) {'+'\n'+
	    '    name = "../output.h5";'+'\n'+
	    '    flippitchsign = yes;'+'\n'+
	    '    time = '+params['time_idx']+';'+'\n'+
            '}'+'\n')
        f.write('@Radiation rad {'+'\n'+
            '    detector = det;'+'\n'+
            '    ignore_trapped = yes;'+'\n'+
            '    ntoroidal   = 7000;' +'\n'+   
            '    model       = cone;' +'\n'+   
            '    output      = image;'+'\n'+ 
            '}'+'\n')
        f.write('@Orbits orbit {'+'\n'+
            '    detector = det;'+'\n'+
            '    ntoroidal   = 7000;'+'\n'+ 
            '    output = "orbit_out.h5";'+'\n'+
            '}'+'\n')
        f.write('@Detector det {'+'\n'+
            '    aperture     = 1.4e-3;'+'\n'+
            '    position     = -0.886, -4.002, -0.332;'+'\n'+
            '    direction    = -0.503, 0.864, -0.01;'+'\n'+
            '    vision_angle = 0.523 fov;'+'\n'+
            '    spectrum     = 3.0e-6,3.5e-6,40;' +'\n'+
            '}'+'\n')
        f.write('@RadiationModel cone (cone) {'+'\n'+
            '    emission = synchrotron;'+'\n'+            
            '}'+'\n')
        f.write('@RadiationOutput image (image) {'+'\n'+
            '    pixels = 600;'+'\n'+ 
            '    output = "image_out.h5";'+'\n'+
            '}'+'\n')
        f.close()
        print(f"SOFT input written to: {input_path}.")
       

    def read_output_file(self, params: dict, run_dir: str):
        """
        Reads the input and output files from the run directory

        Args:
            params (dict): Dictionary containing the input parameters.
            run_dir (str): Directory where the output file is located.

        Returns:
            np.array: Containing the SOFT output image.

        Raises:
            FileNotFoundError: If the output file does not exist.

        """
        file_name_output = os.path.join(run_dir, "image_out.h5")
        if not os.path.exists(file_name_output):
            raise FileNotFoundError(f"{file_name_output}")
        output = h5py.File(file_name_output)
        return output
