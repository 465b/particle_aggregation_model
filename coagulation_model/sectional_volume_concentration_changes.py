# from coagulation_model.particle_size_distribtuion import particle_size_distribution
# from coagulation_model.sectional_coagulation_kernels import sectional_coagulation_kernels
import numpy as np

class SectionalVolumeConcentrationChanges():
    """
    This class calculates the "sectional volume concentration changes" (dQ_{i}/dt)
    as presented in Eq. 9 in Jackson and Lochman (J&L) (10.4319/lo.1992.37.1.0077).
    In Jackson and Lochman they are expressed in terms of mass per volume,
    here we express it in terms of volume per volume.

    (See doc/figures/jl_eq9.png)
    """


    def __init__(self, sectional_coagulation_kernels, particle_size_distribution):
        
        self.sectional_coagulation_kernels = sectional_coagulation_kernels # beta(r_i, r_j)
        
        beta = sectional_coagulation_kernels.data
        self.b1 = beta[0,:,:]
        self.b25 = beta[1,:,:] - beta[2,:,:] - beta[3,:,:] - beta[4,:,:]

        # particle size distribution
        self.particle_size_distribution = particle_size_distribution

        self.data = np.zeros((self.particle_size_distribution.number_size_classes))

        self.components = np.zeros((5,self.particle_size_distribution.number_size_classes))


    def _calculate_volume_concentration_changes_jl(self):
        """
        BROKEN: THIS CONTAINS SOME INDEXING ERROR AND CREATES INCORRECT RESULTS
        -----------------------------------------------------------------------
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
            for ii in range(max(0,ll-1),ll+1):
                for jj in range(max(0,ll-1),ll+1):
                    term_1 += beta[0,ii,jj] * Q[ii] * Q[jj]
            term_1 = 0.5 * term_1

            # term 2.1 (beta2, positive part - j&l; beta2 - burd)           
            term_2_1 = 0
            for ii in range(ll+1):
                term_2_1 += beta[1,ii,ll] * Q[ii]
            term_2_1 = Q[ll] * term_2_1

            # term 2.2 (beta2, negative part - j&l; beta3 - burd)
            term_2_2 = 0
            for ii in range(ll+1):
                term_2_2 += beta[2,ii,ll] * Q[ii]
            term_2_2 = Q[ll] * term_2_2

            # term 3 (beta3 - j&l; beta4 - burd)
            term_3 = 0.5 * beta[3,ll,ll] * Q[ll]**2

            # term 4 (beta4 - j&l; beta5 - burd)
            term_4 = 0
            for ii in range(ll+1,len(self.data)):
                term_4 += beta[4,ii,ll] * Q[ii]
            term_4 = Q[ll] * term_4

            self.components[0,ll] = term_1
            self.components[1,ll] = term_2_1
            self.components[2,ll] = term_2_2
            self.components[3,ll] = term_3
            self.components[4,ll] = term_4


            # print(term_1, term_2_1, term_2_2, term_3, term_4)

            # sum all terms
            self.data[ll] = term_1 - term_2_1 + term_2_2 - term_3 - term_4


    def calc_volume_concentration_changes(self,volume_concentration):
        """
        Calculate the derivatives in the population balance equations.
        Translated from Burds MATLAB code.
        """
            
        Q = volume_concentration
        # Convert to row vector
        Q_r = Q.reshape(1, -1)
        
        # Create shifted vector with 0 padding at start
        Q_shift = np.zeros_like(Q_r)
        Q_shift[0, 1:] = Q_r[0, :-1]
        
        # Calculate terms
        term1 = Q_r @ self.b25  # Matrix multiplication
        term1 = Q_r * term1   # Element-wise multiplication
        
        term2 = Q_r @ self.b1   # Matrix multiplication
        term2 = term2 * Q_shift  # Element-wise multiplication
        
        # Calculate final result and convert back to column vector
        dv_dt = (term1 + term2).T
        
        return dv_dt