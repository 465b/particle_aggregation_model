import numpy as np
from scipy.integrate import dblquad

from particle_size_distribution import set_up_parcile_size_classes

class SectionalCoagulationKernels:
    """
    This class calculates the "sectional coagulation kernels" (\beta_{ijl})
    as presented in the Jackson and Lochman (J&L) (10.4319/lo.1992.37.1.0077)

    (See Fig 'doc/jl_sectional_kernel.png')

    J&L defined the sectional coagulation kernel as beta 1-4 in table 1.
    We split the first sectional kernel into two parts similiar to Burds
    (10.1002/jgrc.20255) matlab model.`
    Hence, 
    
    |   J&L     | Burd
    |  beta_1.1 | beta_1
    |  beta_1.2 | beta_2
    |  beta_2   | beta_3
    |  beta_3   | beta_4
    |  beta_4   | beta_5

    J&L express their calculation based on particle masses.
    We use particle radii instead.
    During calculation we 

    """


    def __init__(self, coagulation_kernel,particle_size_distribution,
                 particle_density = 1, stickiness = 1):
            
            self.coagulation_kernel = coagulation_kernel # beta(r_i, r_j)
            self.particle_size_distribution = particle_size_distribution

            self.particle_density = particle_density
            self.stickiness = stickiness
      
            self.data = np.zeros(
                 (6, # number of sectional kernels + 1
                  self.particle_size_distribution.number_size_classes, # l
                  self.particle_size_distribution.number_size_classes  # i 
                  )
                 )

            

    def symetric_integrand(self,r_i, r_j):

        vol_i = self.coagulation_kernel.volume_sphere(r_i)
        vol_j = self.coagulation_kernel.volume_sphere(r_j)

        beta = self.coagulation_kernel

        integrand = beta.evaluate_kernel(r_i, r_j)
        integrand = integrand * (vol_i + vol_j)/vol_i/vol_j
        
        return integrand


    def asymetric_1_integrand(self,r_i, r_j):

        vol_i = self.coagulation_kernel.volume_sphere(r_i)
        vol_j = self.coagulation_kernel.volume_sphere(r_j)

        beta = self.coagulation_kernel

        integrand = beta.evaluate_kernel(r_i, r_j)
        integrand = integrand / vol_i
        
        return integrand

    
    def asymetric_2_integrand(self,r_i, r_j):

        vol_i = self.coagulation_kernel.volume_sphere(r_i)
        vol_j = self.coagulation_kernel.volume_sphere(r_j)

        beta = self.coagulation_kernel

        integrand = beta.evaluate_kernel(r_i, r_j)
        integrand = integrand / vol_i
        
        return integrand

    
    def sectional_kernel_1_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_1.1 
        """

        number_size_classes = self.particle_size_distribution.number_size_classes

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres
        # for the coagulation calculations we assume that they interact as fractals
        radii = self.coagulation_kernel.radius_fractal(self.coagulation_kernel.volume_sphere(radii))

        # ll corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(2,number_size_classes):
            for ii in range(1,ll): # range goes up to ll-1

                result, error = dblquad(self.symetric_integrand, 
                                        radii[ii-1], 
                                        radii[ii], 
                                        lambda x: np.max([radii[ii-1] - x, radii[ll]]),
                                        radii[ll-1])

                result = (self.stickiness /radii[ii-1] /radii[ll-2]) * result

                self.data[1,ll,ii] = result


    def sectional_kernel_2_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_1.2 
        """

        number_size_classes = self.particle_size_distribution.number_size_classes

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres
        # for the coagulation calculations we assume that they interact as fractals
        radii = self.coagulation_kernel.radius_fractal(self.coagulation_kernel.volume_sphere(radii))

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(2,number_size_classes):
            for ii in range(ll-1,ll): # range goes up to ll-1

                result, error = dblquad(self.symetric_integrand, 
                                        radii[ll-2], 
                                        radii[ll-1], 
                                        radii[ll-2],
                                        radii[ll-1])

                result = (self.stickiness /radii[ll-2] /radii[ll-2]) * result

                self.data[2,ll,ii] = result


    def sectional_kernel_3_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_2
        """

        number_size_classes = self.particle_size_distribution.number_size_classes

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres
        # for the coagulation calculations we assume that they interact as fractals
        radii = self.coagulation_kernel.radius_fractal(self.coagulation_kernel.volume_sphere(radii))

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(2,number_size_classes):
            for ii in range(1,ll): # range goes up to ll-1

                # positive term
                result_1, error = dblquad(self.asymetric_1_integrand, 
                                        radii[ii-1], 
                                        radii[ii],
                                        lambda x: radii[ll]-x,
                                        radii[ll])

                result_1 = (self.stickiness /radii[ii-1] /radii[ll-1]) * result_1

                # negative term
                result_2, error = dblquad(self.asymetric_2_integrand, 
                                        radii[ii-1],
                                        radii[ii],
                                        radii[ll-1],
                                        lambda x: radii[ll]-x)

                result_2 = (self.stickiness /radii[ii-1] /radii[ll-1]) * result_2

                result = result_1 - result_2

                self.data[3,ll,ii] = result


    def sectional_kernel_4_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_3
        """

        number_size_classes = self.particle_size_distribution.number_size_classes

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres
        # for the coagulation calculations we assume that they interact as fractals
        radii = self.coagulation_kernel.radius_fractal(self.coagulation_kernel.volume_sphere(radii))

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(1,number_size_classes):
            for ii in range(ll,ll+1): # ii = ll

                # positive term
                result_1, error = dblquad(self.symetric_integrand, 
                                        radii[ll-1], 
                                        radii[ll],
                                        radii[ll-1],
                                        lambda x: x)

                result_1 = (self.stickiness /radii[ll-1] /radii[ll-1]) * result_1

                # positive term
                result_2, error = dblquad(self.symetric_integrand, 
                                        radii[ll-1],
                                        radii[ll],
                                        lambda x: x,
                                        radii[ll])

                result_2 = (self.stickiness /radii[ll-1] /radii[ll-1]) * result_2

                result = result_1 - result_2

                self.data[4,ll,ii] = result


    def sectional_kernel_5_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_4
        """

        number_size_classes = self.particle_size_distribution.number_size_classes

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres
        # for the coagulation calculations we assume that they interact as fractals
        radii = self.coagulation_kernel.radius_fractal(self.coagulation_kernel.volume_sphere(radii))

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(1,number_size_classes-1): # l < i hence l != number_size_classes
            for ii in range(ll,number_size_classes): # l < i < number_size_classes

                result, error = dblquad(self.asymetric_1_integrand, 
                                        radii[ii-1], 
                                        radii[ii],
                                        radii[ll-1],
                                        radii[ll])

                result = (self.stickiness /radii[ii-1] /radii[ll-1]) * result

                self.data[5,ll,ii] = result


    def eval_all_kernels(self):
        self.sectional_kernel_1_eval()
        self.sectional_kernel_2_eval()
        self.sectional_kernel_3_eval()
        self.sectional_kernel_4_eval()
        self.sectional_kernel_5_eval()
    