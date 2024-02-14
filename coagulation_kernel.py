import numpy as np

class coagulation_kernel(object):
    def __init__(self, list_of_applied_kernels, settling_function):

        self.selected_kernels = []
        for item in list_of_applied_kernels:
            method = getattr(self, item, None)
            if method:
                # If the method exists, call it
                self.selected_kernels.append(method)
            else:
                print(f"No method named '{item}' found.")

        
        # get settling function with the same name in this class and bind it
        method = getattr(self, settling_function, None)
        if method:
            # If the method exists, call it
            self.settling_function = method
        else:
            print(f"No method named '{settling_function}' found.")

        self.set_up_constants()


    def set_up_constants(self):

        param = {

 
        #     'n_sections': 20,  # Number of sections
        #     'n1': 100,  # No. particles cm^{-3} in first section
        #     'temp': 20 + 273,  # Temperature [K]
        #     'alpha': 1.0,  # Stickiness
        #     'dz': 65,  # Layer thickness [m]
        #     'gamma': 0.1,  # Average shear rate [s^{-1}]
        #     'growth': 0.15,  # Specific growth rate in first section [d^{-1}]
        #     'gro_sec': 4,  # Section at which growth in aggregates starts
        #     'num_1': 10**3,  # Number of particle cm^{-3} in first section
        #     'c3': 0.2,  # Disaggregation parameter
        #     'c4': 1.45,  # Disaggregation parameter
        #     't_init': 0.0,  # Initial time for integrations [d]
        #     't_final': 30.0,  # Final time for integrations [d]
        #     'delta_t': 1.0,  # Time interval for output [d]
        }

        # # Derived parameters
        # param['dvisc'] = param['kvisc'] * param['rho_fl']  # Dynamic viscosity [g cm^{-1} s^{-1}]
        # param['del_rho'] = (4.5 * 2.48) * param['kvisc'] * param['rho_fl'] / param['g'] * (param['d0'] / 2) ** (-0.83)
        # param['conBr'] = 2.0 / 3.0 * param['k'] * param['temp'] / param['dvisc']
        # param['v0'] = (np.pi / 6) * param['d0'] ** 3
        # param['v_lower'] = param['v0'] * 2. ** np.arange(param['n_sections'])
        # param['v_upper'] = 2.0 * param['v_lower']
        # param['av_vol'] = 1.5 * param['v_lower']
        # param['dcomb'] = (param['v_lower'] * 6 / np.pi) ** (1.0 / 3.0)
        # param['dwidth'] = (2 ** (1.0 / 3.0) - 1) * param['dcomb']


        # general constants
        # -----------------

        self.gravitational_acceleration = 9.8 #[m/s**2]
        #     'day_to_sec': 8.64e04,  # Seconds in a day [s d^{-1}]
        #     'k': 1.3e-16,  # Boltzmann's constant [erg K^{-1}]


        # suspending liquid properties
        # ----------------------------
        self.density_liquid = 1028  #[kg m^{-3}]  
        self.kinematic_viscostiy = 1e-6  # Kinematic viscosity [m^2 s^{-1}]
        self.dynamic_viscosity = self.kinematic_viscostiy * self.density_liquid  # Dynamic viscosity [kg m^{-1} s^{-1}]

        # general particulate contstants
        # ------------------------------
        self.density_particulate = 2480 # kg/m^3
        self.density_particulate_kg_cm = self.density_particulate*1e-3 # g/cm^3

        # spherical particulate constants
        # -------------------------------
        self.stokes_sphere_settling_const =  (2/9) * self.gravitational_acceleration * (self.density_particulate - self.density_liquid) / self.dynamic_viscosity
        
        # fractal particulate constants
        # -----------------------------

        # For some of the detail see Stemmann et al (2004)
        self.radius_of_sphere_to_radius_of_gyration = 1.36
        
        self.radius_unit_particle_m = 1e-6
        self.radius_unit_particle_cm = self.radius_unit_particle_m * 100 # 1e-4
    
        self.particle_fractal_dimension = 2.33

        amfrac = (4/3 * np.pi) ** (-1 / self.particle_fractal_dimension) * self.radius_unit_particle_m ** (1 - 3 / self.particle_fractal_dimension)
        self.amfrac = amfrac * np.sqrt(0.6)
        self.bmfrac = 1. / self.particle_fractal_dimension

        self.fractal_settling_constant_cm = self.density_particulate_kg_cm * (self.radius_unit_particle_cm)**(-0.83)
        self.fractal_setling_constant = self.fractal_settling_constant_cm * 1000 * 1000 / 100 # SI


    # radius calculation functions
    # ---------------------------
        
    # Following Adrian Burds naming scheme and equations:

    def radius_conserved_volume(self, volume):
        """
        Calculate the radius of a ball assuming a homogoneous volume distribution.

        Parameters:
        - volume: volume of particle
        """

        # by my calculation this should rather be 
        # ( (3/(4 * np.pi)) * volume / density ) ** (1/3)
        # I am particularly confused why there isn't any density in the equation

        return (3/(4 * np.pi) * volume) ** (1/3)


    def radius_fractal(self, volume):
        """
        Calculate the radius of a fractal particle.
        Partly based on Stemmann et al (2004).
        """

        alpha = self.amfrac 
        beta = self.bmfrac

        return alpha * volume ** beta

    def radius_gyration(self, volume):
        """
        Calculate the radius of gyration of a particle.
        """

        return self.radius_of_sphere_to_radius_of_gyration * self.radius_conserved_volume(volume)



    # kernel functions
    # ----------------

    # @staticmethod
    def curviliniar_shear(self,radius_i, radius_j):
        """
        Calculate kernel curvature.

        Parameters:
        - r: Numpy array of particle radii [m]
        - param: Dictionary with key 'r_to_rg' for converting radius to radius of gyration

        Returns:
        - beta: Kernel curvature
        """

        radius_gyration = (radius_i + radius_j) * self.radius_of_sphere_to_radius_of_gyration
        particle_ratio = np.min(radius_i,radius_j) / np.max(radius_i,radius_j)
        coag_efficiency = 1 - (1 + 5*particle_ratio + 2.5*particle_ratio**2) / (1 + particle_ratio)**5
        beta = np.sqrt(8* np.pi / 15) * coag_efficiency * radius_gyration**3

        return beta


    def rectilinear_shear(self,radius_i, radius_j):

        # Calculate radius of gyration
        radius_gyration = (radius_i + radius_j) * self.radius_of_sphere_to_radius_of_gyration
        # # Calculate the kernel value
        beta = 1.3 * radius_gyration**3
        
        return beta

    def rectilinear_differential_sedimentation(self,radius_i, radius_j):
        
        radius_gyration = (radius_i + radius_j) * self.radius_of_sphere_to_radius_of_gyration
        
        velocity_i = self.settling_function(radius_i)
        veloctiy_j = self.settling_function(radius_j)

        # Calculate the kernel value based on the difference in settling velocities and the radius of gyration
        beta = np.pi * abs(velocity_i - veloctiy_j) * radius_gyration**2
        return beta



    # settling functions
    # ------------------


    def stokes_sphere_settling_velocity(self, radius_sphere):
        """
        Calculates the settling velocities of particles of
        given sizes based assuming a perfect homogeniously dense sphere.
        """
        # stokes_sphere_settling_const = (2/9) * self.gravitational_acceleration * (self.density_particulate - self.density_liquid) / self.dynamic_viscosity
        v = self.stokes_sphere_settling_const * radius_sphere**2

        return v


    def stokes_fractal_settling_velocity(self, radius_sphere):
        """
        Settling Velocity calculates the settling velocities of particles of
        given sizes based on the fractal dimension of the particles.
        """

        v = self.fractal_setling_constant * radius_sphere**2

        return v

    def evaluate_kernel(self, radius_i, radius_j):
        """
        Evaluate the kernel for two particles of given sizes.

        Parameters:
        - radius_i: Radius of particle i
        - radius_j: Radius of particle j

        Returns:
        - beta: Kernel value
        """

        beta = 0
        for kernel in self.selected_kernels:
            beta += kernel(radius_i, radius_j)

        return beta
        
    
# shear_stokes = coagulation_kernel(list_of_applied_kernels=['rectilinear_shear'], settling_function='stokes_fractal_settling_velocity')

# shear_stokes.evaluate_kernel(1e-3,1e-3)


