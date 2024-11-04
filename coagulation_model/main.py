from coagulation_model.sectional_volume_concentration_changes import SectionalVolumeConcentrationChanges
from coagulation_model.sectional_coagulation_kernels import SectionalCoagulationKernels
from coagulation_kernel import CoagulationKernel as kernel
from particle_size_distribution import ParticleSizeDistribution as psd

import numpy as np
from datetime import timedelta

particle_size_distribution = psd(radius_sphere_min = 1e-6,radius_sphere_max = 1.1e-4,
                                 type='log10_decline',kwargs={'initial_volume_concentration': 6.283e-6})

coagulation_kernel = kernel(
    list_of_applied_kernels=[
        # 'rectilinear_shear',
        'rectilinear_differential_sedimentation'],
    settling_function='settling_velocity_jackson_lochmann_fractal'
    )

def read_beta_files(num_files, directory="coag_model_adrian_matlab"):
    """
    Read beta_i.txt files into a 3D numpy array.
    
    Args:
        num_files (int): Number of beta files to read (beta_1.txt to beta_n.txt)
        directory (str): Directory containing the beta files
        
    Returns:
        numpy.ndarray: 3D array of shape (num_files, 20, 20) containing all beta matrices
    """
    # Initialize 3D array to store all beta matrices
    beta_matrices = np.zeros((num_files, 20, 20))
    
    # Read each beta file
    for i in range(num_files):
        file_path = f"{directory}/beta_{i+1}.txt"
        try:
            beta_matrices[i] = np.loadtxt(file_path,delimiter=',')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return beta_matrices

matlab_betas = read_beta_files(5)

sectional_kernel = SectionalCoagulationKernels(coagulation_kernel,particle_size_distribution)
sectional_kernel.data = matlab_betas


sectional_mass_changes = SectionalVolumeConcentrationChanges(sectional_kernel,particle_size_distribution)

# t_max 30days in seconds 
t_max = timedelta(days=1).total_seconds()
dt = timedelta(seconds=1).total_seconds()

particle_size_distribution.perform_time_evolution(
    sectional_mass_changes, 
    t_max=t_max, 
    dt=dt, 
    integration_method='LSODA'
    )




