# from coagulation_model.particle_size_distribtuion import particle_size_distribution
# from coagulation_model.sectional_coagulation_kernels import sectional_coagulation_kernels
import numpy as np

class SectionalMassChanges():
    """
    This class calculates the "sectional mass changes" (dQ_{i}/dt)
    as presented in Eq. 9 in Jackson and Lochman (J&L) (10.4319/lo.1992.37.1.0077)

    (See doc/figures/jl_eq9.png)
    """


    def __init__(self, sectional_coagulation_kernels, particle_size_distribution):
        
        self.sectional_coagulation_kernels = sectional_coagulation_kernels # beta(r_i, r_j)
        self.particle_size_distribution = particle_size_distribution

        self.data = np.zeros((self.particle_size_distribution.number_size_classes))

    def calculate_mass_changes(self):
        """
        This function calculates the mass changes for
        each size class i at time t.
        It corresponds to eq. 9 in J&L.
        (doc/figures/jl_eq9.png)
        """

        # sectional kernels
        beta = self.sectional_coagulation_kernels.data
        # particle mass in each size class
        # corresponds to Q_i in eq. 9 in J&L
        Q = self.particle_size_distribution.data

        for ll in range(len(self.data)):

            # term 1
            term_1 = 0 # for j<l: beta_1 & beta 2 = 0

            # term 2           
            term_2_sum = beta[3,ll]*Q
            term_2_sum = np.sum(term_2_sum[:ll])

            term_2 = Q[ll] * term_2_sum

            # term 3
                        # self.data[2,ll,ii] = result
            term_3 = 0.5 * beta[4,ll,ll] * Q[ll]**2

            # term 4
            term_4_sum = beta[5,ll]*Q
            term_4_sum = np.sum(term_4_sum[ll+1:])

            term_4 = Q[ll] * term_4_sum

            # sum all terms
            self.data[ll] = term_1 - term_2 - term_3 - term_4

        




