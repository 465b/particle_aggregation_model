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

            # term 1 (beta1 - j&l; beta1 - burd)
            term_1 = 0
            for ii in range(ll):
                for jj in range(ll):
                    term_1 += beta[0,ii,jj] * Q[ii] * Q[jj]
            term_1 = 0.5 * term_1

            # term 2.1 (beta2 - j&l; beta2 - burd)           
            term_2_1 = 0
            for ii in range(ll):
                term_2_1 += beta[1,ii,jj] * Q[ii]
            term_2_1 = Q[ll] * term_2_1

            # term 2.2 (beta2 - j&l; beta3 - burd)
            term_2_2 = 0
            for ii in range(ll):
                term_2_2 += beta[2,ii,jj] * Q[ii]
            term_2_2 = Q[ll] * term_2_2

            # term 3 (beta3 - j&l; beta4 - burd)
            term_3 = 0.5 * beta[3,ll,ll] * Q[ll]**2

            # term 4 (beta4 - j&l; beta5 - burd)
            term_4 = 0
            for ii in range(ll+1,len(self.data)):
                term_4 += beta[4,ii,ll] * Q[ii]
            term_4 = Q[ll] * term_4

            # sum all terms
            self.data[ll] = term_1 - term_2_1 + term_2_2 - term_3 - term_4

            # print(self.data)