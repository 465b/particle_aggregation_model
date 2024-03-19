import numpy as np 

def set_up_parcile_size_classes(radius_sphere_min,radius_sphere_max):
    
    # calculate new number of size classes based on
    # radius_sphere_min and radius_sphere_max
    # with the condition that the volume of each 
    # size class doubles the volume of the previous size class

    # min and max volume of the size classes
    volume_min = (4/3) * np.pi * radius_sphere_min ** 3
    volume_max = (4/3) * np.pi * radius_sphere_max ** 3

    # Calculate the number of times we can double the volume 
    # before exceeding radius_sphere_max
    n = int(np.floor(np.log2(volume_max / volume_min))) + 1
    
    # # Generate the sizes using powers of 2, starting from volume_min

    volume_boundary_spheres = volume_min * np.power(2, np.arange(n))
    radius_boundary_spheres = (volume_boundary_spheres * 3 / (4 * np.pi)) ** (1 / 3)  
        
    radius_mean_spheres = np.convolve(
                  radius_boundary_spheres, 
                  np.ones(2, dtype=int), 'valid'
                  ) / 2

    number_size_classes = len(radius_mean_spheres)

    return number_size_classes, radius_boundary_spheres, radius_mean_spheres


class ParticleSizeDistribution():

    def __init__(self, radius_sphere_min=1e-6, radius_sphere_max=1e-3,
                 type=None, kwargs=None,seed = 0):

        # seeding random number generator for reproducability
        self.seed = np.random.seed(seed)

        self.radius_sphere_min = radius_sphere_min
        self.radius_sphere_max = radius_sphere_max
        
        self.number_size_classes = set_up_parcile_size_classes(
            radius_sphere_min,radius_sphere_max)[0]
        self.radius_boundary_spheres = set_up_parcile_size_classes(
            radius_sphere_min,radius_sphere_max)[1]
        self.radius_mean_spheres = set_up_parcile_size_classes(
            radius_sphere_min,radius_sphere_max)[2]
        

        if type is not None:
            self.type = type
            self.kwargs = kwargs

            self.distribution = self.select_distribution()

            # data containts the particle mass in each bin
            self.data = self.fill_particle_size_classes_with_masses()


    def select_distribution(self):
        if self.type == 'lognormal':
            return self.lognormal(**self.kwargs)
        elif self.type == 'powerlaw':
            return self.powerlaw(**self.kwargs)
        elif self.type == 'bimodal':
            return self.bimodal(**self.kwargs)
        else:
            raise ValueError(f'Distribution type {self.type} not recognized')

    
    def fill_particle_size_classes_with_masses(self):
        """
        This function fills the particle size classes with masses
        based on the particle size distribution.
        """
        
        # number of particles in each size class
        probability_distribution = self.distribution(self.radius_mean_spheres)

        # roll for particle mass based on the probability distribution
        # particle_mass_in_bin = np.random.poisson(probability_distribution)
        particle_mass_in_bin = probability_distribution * 1

        return particle_mass_in_bin


    # def fill_particle_size_classes_with_particles(self):
    #     """
    #     This function fills the particle size classes with particles
    #     based on the particle size distribution.
    #     """
        
    #     # number of particles in each size class
    #     probability_distribution = self.distribution(self.radius_mean_spheres)

    #     # roll for particle count based on the probability distribution
    #     particle_counts_in_bin = np.random.poisson(probability_distribution)

    #     return particle_counts_in_bin
        

    @staticmethod
    def powerlaw(a,k):
        """
        -beta > 1 + 2.28 (sec. 2.2. in https://doi.org/10.1016/S0967-0637(99)00032-1)
        """
        def construct_power_law(x,a=a,k=k):
            y = a * x ** -k
            return y
        
        return construct_power_law





