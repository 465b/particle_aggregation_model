import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

## Plotting particle distributions
# --------------------------------

def plot_particle_size_classes(radius_boundary_spheres,xaxis_label='particle radius [m]'):
    """
    Plots the size class distribution on a number line
    """

    radius_mean_spheres = np.convolve(radius_boundary_spheres, np.ones(2, dtype=int), 'valid') / 2

    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10,1))
    
    n_points = len(radius_boundary_spheres)
    ax.scatter(radius_boundary_spheres[:n_points], np.zeros(n_points), 
              color='red', marker='|', s=400,label='class boundary')
    ax.scatter(radius_mean_spheres[:n_points-1], np.zeros(n_points-1), 
              color='blue', marker='|', s=300,label='class mean')
    
    # Format plot
    ax.set_yticks([])
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))

    # add legend
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=2, borderaxespad=0.)
    
    # show number of classes
    ax.text(1, +1, f'{n_points-1} classes', transform=ax.transAxes,
            ha='right', va='center', fontsize=12)

    # add labels
    ax.set_xlabel(xaxis_label)
    
    # return fig
    return fig, ax


def plot_particle_size_distribution(particle_size_distribution, x_axis_scale='log', y_axis_scale='linear'):
    """
    Plots the particle size distribution as a bar chart
    using the particle_size_distribution mean radius
    """
    
    radius_mean_spheres = np.convolve(particle_size_distribution.radius_boundary_spheres, np.ones(2, dtype=int), 'valid') / 2

    n_classes = len(radius_mean_spheres)
    fig, ax = plt.subplots()
    # each bar is centered on the mean radius[ii] of the size class
    # extending from radius_boundary_spheres[ii] to radius_boundary_spheres[ii+1]
    if hasattr(particle_size_distribution, 'data'):
        data_to_plot = particle_size_distribution.data
    else:
        data_to_plot = particle_size_distribution.initial_volume_concentration
    ax.bar(radius_mean_spheres, data_to_plot,
           width=np.diff(particle_size_distribution.radius_boundary_spheres))
    ax.set_xlabel("particle size class radius")
    ax.set_ylabel('total something? in particle class ]')
    ax.set_title('Particle size distribution')

    ax.set_xscale(x_axis_scale)
    ax.set_yscale(y_axis_scale)

    # add text with the number of classes
    ax.text(.9, .9, f'{n_classes} classes', transform=ax.transAxes,
            ha='right', va='center', fontsize=12)

    return fig, ax


## Plotting coagulation kernels
#  ----------------------------

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
                                save_path=None,
                                scale = 'log'):

    betas_test = saturate_kernel_map(test_kernel, n, r_min, r_max)
    betas_reference = saturate_kernel_map(reference_kernel, n, r_min, r_max)
    betas_normalized = betas_test / betas_reference

    norm = LogNorm(vmin=norm_range[0], vmax=norm_range[1])

    fig, ax = plt.subplots(figsize=(10,5))


    image = ax.imshow(betas_normalized, #[m**3/s]
                      extent=np.array([r_min, r_max, r_min, r_max]), # [mm]
                      norm=norm, origin='lower',
                      interpolation='bicubic'
                      )
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label('Beta ratio')
    ax.set_xlabel('Radius of particle i (m)')
    ax.set_ylabel('Radius of particle j (m)')
    ax.set_title(f'{test_kernel_label} / {reference_kernel_label}')

    if scale == 'log':
        ax.set_yscale('log')
        ax.set_xscale('log')

    if save_path:
        plt.savefig(save_path)


def plot_kernel_diff_map(test_kernel, test_kernel_label, 
                                reference_kernel, reference_kernel_label,
                                n=100, r_min=1e-6, r_max=1e-3, 
                                norm_range=[1e-15, 1e-8],
                                save_path=None):
    
    betas_test = saturate_kernel_map(test_kernel, n, r_min, r_max)
    betas_reference = saturate_kernel_map(reference_kernel, n, r_min, r_max)
    betas_diff = betas_test - betas_reference

    norm = LogNorm(vmin=norm_range[0], vmax=norm_range[1])

    fig, ax = plt.subplots(figsize=(10,5))

    image = ax.imshow(betas_diff, #[m**3/s]
                      extent=np.array([r_min, r_max, r_min, r_max]), # [mm]
                      norm=norm, origin='lower',
                      interpolation='bicubic'
                      )
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label('Beta (m^3/s)')
    ax.set_xlabel('Radius of particle i (m)')
    ax.set_ylabel('Radius of particle j (m)')
    ax.set_title(f'{test_kernel_label} - {reference_kernel_label}')

    if save_path:
        plt.savefig(save_path)


def plot_kernel_map(kernel, kernel_label,
                    n=100, r_min=1e-6, r_max=1e-3,
                    norm_range=[1e-15, 1e-8],
                    scale = 'log'):
    
    betas = saturate_kernel_map(kernel, n, r_min, r_max)

    fig, ax = plt.subplots()

    norm = LogNorm(vmin=norm_range[0], vmax=norm_range[1])

    # imshow with log scale with a norm range
    image = ax.imshow(betas,
                      extent=np.array([r_min, r_max, r_min, r_max]), # [mm]
                      norm=norm, origin='lower',
                      interpolation='bicubic'
                      )

    # colorbar 
    cbar = plt.colorbar(image, ax=ax)
    # add colobar label
    cbar.set_label('Beta (m^3/s)')
    ax.set_xlabel('Radius of particle i (m)')
    ax.set_ylabel('Radius of particle j (m)')

    if scale == 'log':
        ax.set_yscale('log')
        ax.set_xscale('log')

    # add title
    fig.suptitle(kernel_label)

    return fig, ax


## Plotting sectional coagulation kernels
# --------------------------------------

def plot_sectional_kernels(sectional_kernel_data, figsize=(20, 4)):
    """
    Plot sectional coagulation kernels side by side.
    
    Parameters:
    -----------
    sectional_kernel_data : numpy.ndarray
        Array containing the sectional kernel data with shape (n_kernels, n_rows, n_cols)
        or (n_rows, n_cols)
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (20, 4)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    axs : numpy.ndarray
        Array of subplot axes
    """
    # Handle both 2D and 3D arrays
    if len(sectional_kernel_data.shape) == 2:
        sectional_kernel_data = sectional_kernel_data[:,np.newaxis,:]
        
    n_kernels = sectional_kernel_data.shape[0]
    fig, axs = plt.subplots(1, n_kernels, figsize=figsize)
    
    # Convert axs to array if only one kernel
    if n_kernels == 1:
        axs = np.array([axs])
    
    # Find global min and max for consistent colorbar
    vmin = np.min(sectional_kernel_data[sectional_kernel_data > 0])  # Exclude zeros
    vmax = np.max(sectional_kernel_data)
    norm = LogNorm(vmin=vmin, vmax=vmax)
        
    for i in range(n_kernels):
        im = axs[i].imshow(sectional_kernel_data[i], 
                          cmap='viridis', 
                          origin='lower',
                          norm=norm)
        axs[i].set_title(f'Sectional Kernel {i+1}')
        axs[i].set_xlabel('l')
        axs[i].set_ylabel('i')
        fig.colorbar(im, ax=axs[i])
    
    plt.tight_layout()
    return fig, axs