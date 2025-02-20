from TPED.projects.instability_analysis.src.fingerprints import fingerprint_quantities, fingerprint_to_csv

# path can be either a single directory (str) or a list of directories
scan_path = '/pscratch/sd/j/joeschm/NSXTU_discharges/132588/r_0.736_q4_MTM_mode/convergence_check/nz0_hpyz_edgeopt_scans/nz0_1024_edgeopt_04'
scan_path = ['path/to/scanfiles0000', 'path/to/scanfiles0001', 'path/to/scanfiles0002']

f_list = fingerprint_quantities(scan_path) 

output_dir = 'path/to/output/dir'
fingerprint_to_csv(f_list, output_dir) # will save to output_dir


fingerprint_to_csv(f_list) # will save to current working directory