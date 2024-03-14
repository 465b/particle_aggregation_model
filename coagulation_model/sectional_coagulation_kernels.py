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
    |  beta_1.2 | beta_1
    |  beta_2.1 | beta_2  <- positive, first term
    |  beta_2.2 | beta_3  <- negative, second term
    |  beta_3   | beta_4  <- both integrals put together
    |  beta_4   | beta_5

    J&L express their calculation based on particle masses.
    We use particle radii instead.
    During calculation we 

    """


    def __init__(self, coagulation_kernel,particle_size_distribution,
                 particle_density = 1, stickiness = 1, debug = False):
            
            self.coagulation_kernel = coagulation_kernel # beta(r_i, r_j)
            self.particle_size_distribution = particle_size_distribution

            self.particle_density = particle_density
            self.stickiness = stickiness
      
            self.data = np.zeros(
                 (5, # number of sectional kernels (beta_1 - beta_5)
                  self.particle_size_distribution.number_size_classes, # i
                  self.particle_size_distribution.number_size_classes  # l 
                  )
                 )

            self.debug = debug
            

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
        integrand = integrand / vol_j
        
        return integrand

    
    def sectional_kernel_1_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_1.1 
        """
        if self.debug:
            print('Calculating sectional beta 1')

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres

        # ll corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(2,len(radii)):
            jj = ll - 1
            for ii in range(1,ll): # range goes up to ll-1

                if self.debug:
                    print(f'Calculating beta_1 for i={ii} and l={ll}')

                result, error = dblquad(self.symetric_integrand, 
                                        radii[ii-1], 
                                        radii[ii], 
                                        lambda x: np.max([radii[jj-1] - x, radii[jj-1]]),
                                        radii[jj-1])

                result = (self.stickiness /radii[ii-1] /radii[jj-1]) * result

                # 1 -1 to match with the matlab one based indexing
                self.data[1 -1,ii-1,ll-1] = result

                # print self data in a nice 9x9 matrix well formatted

                if self.debug:
                    print(
                        np.array2string(
                            result, precision=2, 
                            separator=' ', suppress_small=True
                            )
                          )



    def sectional_kernel_2_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_1.2 
        """

        if self.debug:
            print('Calculating sectional beta 2')

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(2,len(radii)):
            for ii in range(1,ll): # range goes up to ll-1

                if self.debug:
                    print(f'Calculating beta_2 for i={ii} and l={ll}')                

                result, error = dblquad(self.asymetric_1_integrand, 
                                        radii[ii-1], 
                                        radii[ii], 
                                        lambda x: radii[ll]-x,
                                        radii[ll])

                result = (self.stickiness /radii[ii-1] /radii[ll-1]) * result

                self.data[2 -1,ii-1,ll-1] = result

                if self.debug:
                    print(
                        np.array2string(
                            result, precision=2, 
                            separator=' ', suppress_small=True
                            )
                          )                


    def sectional_kernel_3_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_2
        """

        if self.debug:
            print('Calculating sectional beta 3')

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(2,len(radii)):
            for ii in range(1,ll): # range goes up to ll-1

                if self.debug:
                    print(f'Calculating beta_3 for i={ii} and l={ll}')            

                # positive term
                result, error = dblquad(self.asymetric_2_integrand, 
                                        radii[ii-1], 
                                        radii[ii],
                                        radii[ll-1],
                                        lambda x: radii[ll]-x)

                result = (self.stickiness /radii[ii-1] /radii[ll-1]) * result

                self.data[3 -1,ii-1,ll-1] = result

                if self.debug:
                    print(
                        np.array2string(
                            result, precision=2, 
                            separator=' ', suppress_small=True
                            )
                          )                


    def sectional_kernel_4_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_3
        """

        if self.debug:
            print('Calculating sectional beta 4')

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(1,len(radii)):
            for ii in range(ll,ll+1): # ii = ll

                if self.debug:
                    print(f'Calculating beta_4 for i={ii} and l={ll}')

                # positive term
                result, error = dblquad(self.symetric_integrand, 
                                        radii[ii-1], 
                                        radii[ii],
                                        radii[ll-1],
                                        radii[ll])

                result = (self.stickiness /radii[ii-1] /radii[ll-1]) * result

                self.data[4 -1,ii-1,ll-1] = result

                if self.debug:
                    print(
                        np.array2string(
                            result, precision=2, 
                            separator=' ', suppress_small=True
                            )
                          )                


    def sectional_kernel_5_eval(self):
        """
        Calculates the first sectional coagulation kernel beta_1
        for all size classes.
        See J%L beta_4
        """

        if self.debug:
            print('Calculating sectional beta 5')

        # by default particles are assumed to be perfect spheres
        radii = self.particle_size_distribution.radius_boundary_spheres

        # ii corresponds to 'l' in J&l (doc/jl_sectional_kernel.png)
        # number_size_clases corresponds to 's'
        for ll in range(1,len(radii)): # l < i hence l != number_size_classes
            for ii in range(ll+1,len(radii)): # l < i < number_size_classes

                if self.debug:
                    print(f'Calculating beta_5 for i={ii} and l={ll}')

                result, error = dblquad(self.asymetric_1_integrand, 
                                        radii[ii-1], 
                                        radii[ii],
                                        radii[ll-1],
                                        radii[ll])

                result = (self.stickiness /radii[ii-1] /radii[ll-1]) * result

                self.data[5 -1,ii-1,ll-1] = result

                if self.debug:
                    print(
                        np.array2string(
                            result, precision=2, 
                            separator=' ', suppress_small=True
                            )
                          )                


    def eval_all_kernels(self):
        self.sectional_kernel_1_eval()
        self.sectional_kernel_2_eval()
        self.sectional_kernel_3_eval()
        self.sectional_kernel_4_eval()
        self.sectional_kernel_5_eval()
    