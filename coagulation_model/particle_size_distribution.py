import numpy as np 



class ParticleSizeDistribution():

    def __init__(self, radius_sphere_min=1e-6, radius_sphere_max=1e-3,
                 type=None, kwargs=None,seed = 0):

        # seeding random number generator for reproducability
        self.seed = np.random.seed(seed)

        self.radius_sphere_min = radius_sphere_min
        self.radius_sphere_max = radius_sphere_max
        
        self.number_size_classes = self._set_up_particle_size_classes()[0]
        self.radius_boundary_spheres = self._set_up_particle_size_classes()[1]
        self.radius_mean_spheres = self._set_up_particle_size_classes()[2]
        self.volume_boundary_spheres = self._set_up_particle_size_classes()[3]
        self.volume_mean_spheres = self._set_up_particle_size_classes()[4]


        if type is not None:
            self.type = type
            self.kwargs = kwargs

            self.distribution = self.select_distribution()

            # data containts the particle volume concentration 
            # so volume of aggregates in each size class per unit volume
            # i.e. relative volume of aggregates in each size class
            # [m^{3} m^{-3}]
            self.data = self.fill_particle_size_classes_with_volumes() 
    
    
    def _set_up_particle_size_classes(self):
        """
        Set up the particle size classes based on the range of radii of the aggregates.

        Returns:
        - number_size_classes: number of size classes
        - radius_boundary_spheres: radius of the boundaries of the size classes [m]
        - radius_mean_spheres: radius of the mean of the size classes [m]
        - volume_boundary_spheres: volume of the boundaries of the size classes [m^3]
        - volume_mean_spheres: volume of the mean of the size classes [m^3]
        """
        
        # min and max volume of the size classes
        volume_min = (4/3) * np.pi * self.radius_sphere_min ** 3 # [m^3]
        volume_max = (4/3) * np.pi * self.radius_sphere_max ** 3 # [m^3]

        # Calculate the number of times we can double the volume 
        # before exceeding radius_sphere_max
        n = int(np.floor(np.log2(volume_max / volume_min))) + 1
        
        # Generate the sizes using powers of 2, starting from volume_min
        volume_boundary_spheres = volume_min * np.power(2, np.arange(n)) # [m^3]
        radius_boundary_spheres = (volume_boundary_spheres * 3 / (4 * np.pi)) ** (1 / 3) # [m]
            
        radius_mean_spheres = np.convolve(
                      radius_boundary_spheres, 
                      np.ones(2, dtype=int), 'valid'
                      ) / 2 # [m]
        
        volume_mean_spheres = (4/3) * np.pi * radius_mean_spheres ** 3 # [m^3]

        number_size_classes = len(radius_mean_spheres)

        return number_size_classes, radius_boundary_spheres, radius_mean_spheres, volume_boundary_spheres, volume_mean_spheres


    def select_distribution(self):
        if self.type == 'lognormal':
            return self.lognormal(**self.kwargs)
        elif self.type == 'log10_decline':
            return self.log10_decline(**self.kwargs)
        elif self.type == 'powerlaw':
            return self.powerlaw(**self.kwargs)
        elif self.type == 'bimodal':
            return self.bimodal(**self.kwargs)
        else:
            raise ValueError(f'Distribution type {self.type} not recognized')

    
    def fill_particle_size_classes_with_volumes(self):
        """
        This function fills the particle size classes with volumes
        based on the particle size distribution.
        """
        
        # volume concentration of particles in each size class [m^3/m^3]
        volume_concentration = self.distribution(self.radius_mean_spheres)

        return volume_concentration

        

    @staticmethod
    def powerlaw(a,k):
        """
        Power law distribution

        Parameters:
        - a: amplitude
        - k: power law exponent

        Returns:
        - power_law_distribution: function
        """
        def construct_power_law(x,a=a,k=k):
            y = a * x ** -k
            return y
        
        return construct_power_law
    
    @staticmethod
    def lognormal(mu, sigma):
        """
        Log normal distribution

        Parameters:
        - mu: mean of the log normal distribution
        - sigma: standard deviation of the log normal distribution

        Returns:
        - lognormal_distribution: function
        """

        def construct_lognormal(x, mu=mu, sigma=sigma):
            y = np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * np.sqrt(2 * np.pi))
            return y
        
        return construct_lognormal
    

    @staticmethod
    def log10_decline(initial_volume_concentration):
        """
        Calculate initial particle size distribution spectrum with decreasing
        volume concentrations by factors of 10 based on Adrian Bur.
        Warning: Volume concentration distribution is independent of the actual size of the particles.

        Parameters:
        - initial_volume_concentration: Initial volume concentration for the first size class

        Returns:
        - y: Array of volume concentrations for each size class [m^3/m^3]
        """

        def construct_log10_decline(x, initial_volume_concentration=initial_volume_concentration):
            # Initialize spectrum with the same volume in first bin

            number_size_classes = len(x)
            y = np.ones(number_size_classes) * initial_volume_concentration
        
            # Create decreasing factors (10^0, 10^1, 10^2, ...)
            decrease_factors = np.power(10.0, np.arange(number_size_classes))
        
            # # Apply decreasing factors and ensure minimum concentration
            y = np.maximum(y / decrease_factors, 1.0e-30)
        
            return y

        return construct_log10_decline
