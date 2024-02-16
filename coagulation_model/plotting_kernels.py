import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def saturate_kernel_map(kernel, n=100, r_min=1e-6, r_max=1e-3):
    
    betas = np.zeros((n,n))

    r_i = np.linspace(r_min, r_max, n)
    r_j = np.linspace(r_min, r_max, n)

    # calculate beta values
    for i in range(n):
        for j in range(n):
            betas[i,j] = kernel.evaluate_kernel(r_i[i], r_j[j])

    return betas



def plot_normalized_kernel_map(test_kernel, test_kernel_label, 
                                reference_kernel, reference_kernel_label,
                                n=100, r_min=1e-6, r_max=1e-3, 
                                norm_range=[1e-15, 1e-8],
                                save_path=None):

    betas_test = saturate_kernel_map(test_kernel, n, r_min, r_max)
    betas_reference = saturate_kernel_map(reference_kernel, n, r_min, r_max)
    betas_normalized = betas_test / betas_reference

    norm = LogNorm(vmin=norm_range[0], vmax=norm_range[1])

    fig, ax = plt.subplots(figsize=(10,5))

    image = ax.imshow(betas_normalized, #[m**3/s]
                      extent=np.array([r_min, r_max, r_min, r_max]), # [mm]
                      norm=norm, origin='lower')
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label('Beta (m^3/s)')
    ax.set_xlabel('Radius of particle i (m)')
    ax.set_ylabel('Radius of particle j (m)')
    ax.set_title(f'{test_kernel_label} / {reference_kernel_label}')

    if save_path:
        plt.savefig(save_path)



    


# create empty matrix to store beta values
def plot_kernel_map(kernel, kernel_label, n=100, r_min=1e-6, r_max=1e-3,norm_range=[1e-15, 1e-8]):
    
    betas = saturate_kernel_map(kernel, n, r_min, r_max)

    fig, ax = plt.subplots()

    norm = LogNorm(vmin=norm_range[0], vmax=norm_range[1])

    # imshow with log scale with a norm range
    image = ax.imshow(betas, extent=np.array([r_min, r_max, r_min, r_max]), norm=norm, origin='lower')

    # colorbar 
    cbar = plt.colorbar(image, ax=ax)
    # add colobar label
    cbar.set_label('Beta (m^3/s)')
    ax.set_xlabel('Radius of particle i (m)')
    ax.set_ylabel('Radius of particle j (m)')

    # add title
    fig.suptitle(kernel_label)

    return fig, ax