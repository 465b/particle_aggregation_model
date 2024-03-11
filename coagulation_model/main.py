from sectional_mass_changes import SectionalMassChanges
from sectional_coagulation_kernels import SectionalCoagulationKernels
from coagulation_kernel import CoagulationKernel as kernel
from particle_size_distribution import ParticleSizeDistribution as psd

coagulation_kernel = kernel(
    list_of_applied_kernels=[
        # 'rectilinear_shear',
        'rectilinear_differential_sedimentation'],
    settling_function='settling_velocity_kriest_porous_aggregate'
    )

particle_size_distribution = psd(radius_sphere_min = 1e-6,radius_sphere_max = 1e-3,
                                 type='powerlaw',kwargs={'a':2e-12,'k':3})

sectional_kernel = SectionalCoagulationKernels(coagulation_kernel,particle_size_distribution)
sectional_kernel.eval_all_kernels()

sectional_mass_changes = SectionalMassChanges(sectional_kernel,particle_size_distribution)

sectional_mass_changes.calculate_mass_changes()

print(sectional_mass_changes.data)




