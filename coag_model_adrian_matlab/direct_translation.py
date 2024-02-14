import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode15s


def SetUpCoag():
    """
    SetUpCoag obtains user options for the coagulation calculations and then
    does some housecleaning

    Returns:
        param (dict): Physical parameters and section/coagulation related parameters
    """

    param = {}

    # Physical parameters
    param['fluid_density'] = 1.0275  # Fluid density [g cm^{-3}]
    param['kinematic_viscosity'] = 0.01  # Kinematic viscosity [cm^2 s^{-1}]
    param['g'] = 980  # Accel. due to gravity [cm s^{-2}]
    param['day_to_sec'] = 8.64e04  # Seconds in a day [s d^{-1}]
    param['k'] = 1.3e-16  # Boltzmann's constant [erg K^{-1}]
    param['r_to_rg'] = 1.36  # Interaction radius to radius of gyration

    # Section/coagulation related parameters
    param['n_sections'] = 20  # Number of sections
    param['kernel'] = 'KernelBrown'  # Kernel type
    param['d0'] = 20e-4  # Diameter of unit particle [cm] [20e-4]
    param['fr_dim'] = 2.33  # Particle fractal dimension
    param['n1'] = 100  # No. particles cm^{-3} in first section

    # Other input parameters
    param['temp'] = 20 + 273  # Temperature [K]
    param['alpha'] = 1.0  # Stickiness
    param['dz'] = 65  # Layer thickness [m]
    param['gamma'] = 0.1  # Average shear rate [s^{-1}]
    param['growth'] = 0.15  # Specific growth rate in first section [d^{-1}]
    param['gro_sec'] = 4  # Section at which growth in aggregates starts
    param['num_1'] = 10**3  # Number of particle cm^{-3} in first section [40]

    # Disaggregation parameters
    param['c3'] = 0.2
    param['c4'] = 1.45

    # Parameters for solving equations
    param['t_init'] = 0.0  # Initial time for integrations [d]
    param['t_final'] = 30.0  # Final time for integrations [d]
    param['delta_t'] = 1.0  # Time interval for output [d]

    # Derived parameters
    param['dvisc'] = param['kinematic_viscosity'] * param['fluid_density']  # Dynamic viscosity [g cm^{-1} s^{-1}]
    param['del_rho'] = (4.5 * 2.48) * param['kinematic_viscosity'] * param['fluid_density'] / param['g'] * (param['d0'] / 2)**(-0.83)

    param['conBr'] = 2.0 / 3.0 * param['k'] * param['temp'] / param['dvisc']

    param['a0'] = param['d0'] / 2
    param['v0'] = (np.pi / 6) * param['d0']**3

    param['v_lower'] = param['v0'] * 2**(np.arange(0, param['n_sections']))
    param['v_upper'] = 2.0 * param['v_lower']

    param['av_vol'] = 1.5 * param['v_lower']
    param['dcomb'] = (param['v_lower'] * 6 / np.pi)**(1.0 / 3.0)
    param['dwidth'] = (2**(1.0 / 3.0) - 1) * param['dcomb']

    amfrac = (4.0 / 3.0 * np.pi)**(-1.0 / param['fr_dim']) * param['a0']**(1.0 - 3.0 / param['fr_dim'])
    param['amfrac'] = amfrac * np.sqrt(0.6)
    param['bmfrac'] = 1.0 / param['fr_dim']

    param['setcon'] = (2.0 / 9.0) * param['del_rho'] / param['fluid_density'] * param['g'] / param['kinematic_viscosity']

    return param



def CalcGrowth(p):
    """
    CalcGrowth calculates the matrix of terms that represent algal cell
    growth in the population balance equation.
    """
    
    growth_loss = np.zeros(p['n_sections'])
    growth_gain = np.zeros(p['n_sections'] - 1)
    
    if p['gro_sec'] > 0:
        growth_loss[p['gro_sec'] - 1 : p['n_sections'] - 1] = -1
        growth_gain[p['gro_sec'] - 1 : ] = 2
    
    growth = np.diag(growth_loss) + np.diag(growth_gain, -1)
    growth[0, 0] = 1
    
    growth = p['growth'] * growth
    return growth


def CalcSinkingLoss(p):
    """
    CalcExport calculates specific loss out of the layer because of particles
    sinking.
    """
    
    fractal_radius = p['amfrac'] * p['av_vol'] ** p['bmfrac']
    conserved_radius = (0.75 / np.pi * p['av_vol']) ** (1.0 / 3.0)
    
    settling_vely = SettlingVelocity(fractal_radius, conserved_radius, p['setcon'])
    
    sink_loss = np.diag(settling_vely) / 100 * p['day_to_sec'] / p['dz']
    
    return sink_loss


def SettlingVelocity(r, rcons, sett_const):
    """
    Settling Velocity calculates the settling velocities of particles of
    given sizes.

    USAGE:
        v = SettlingVelocity(r, rcons, sett_const)

    Parameters:
    - v: particle settling velocities [cm s^{-1}]
    - r: particle radii [cm]
    - rcons: radii of particle conserved volumes [cm]
    - sett_const: (2g/9eta)*(delta_rho/fluid_densityuid)
    """
    
    v = sett_const * rcons**3 / r
    return v


import numpy as np

def CalcInitialSpec(p, p2):
    """
    CalcInitialSpec calculates the initial spectrum. At the present, this is
    based on George's code and calculates a spectrum with equal volume in
    each section.
    """
    
    spec_init = p['av_vol'][0] * np.ones(p['n_sections'])
    
    tfactor = 10. ** np.arange(p['n_sections'])
    
    spec_init = np.maximum(spec_init / tfactor, 1.0e-30)
    
    spec_init = spec_init * p['num_1']
    
    return spec_init


def CalcCoagJac(t, vcon, p2):
    """
    CalcCoagJac calculates the Jacobian for the coagulation equations. Each
    row corresponds to a different function and each column to the derivative
    of that function with respect to a different variable.
    """
    
    n_sections = len(vcon)
    
    vcon_r = vcon.reshape(-1, 1).T
    
    vcon_mat = np.tile(vcon_r, (n_sections, 1))
    vcon_shift = np.hstack((np.zeros((n_sections, 1)), vcon_mat[:, :-1]))
    
    # First calculate the df_i/dy_i terms
    term1 = np.diag(vcon_r.flatten() * p2['b25'])
    term1 += np.diag(vcon_r.flatten()) @ np.diag(p2['b25'])
    
    # Calculate the df_i/dy_{i-1} terms
    term2a = np.diag(vcon_r.flatten() * p2['b1'][1:], -1)
    
    term2b = np.diag(np.diag(p2['b1'], 1).T * vcon_r.flatten()[:-1], -1)
    
    term2c = np.diag(vcon_r.flatten()[1:], -1) * p2['b25'].reshape(-1, 1)
    
    term2 = term2a + term2b + term2c
    
    # Calculate the df_i/dy_j terms (j != i, i-1)
    term3a = p2['b1'] * vcon_shift
    term3b = p2['b25'] * vcon_mat
    term3 = (term3a + term3b).T
    term3 = np.triu(term3, 2) + np.tril(term3, -1)
    
    # Now the linear terms
    lin_term = p2['linear']
    
    dfdy = term1 + term2 + term3 + lin_term - p2['disagg_minus'] + p2['disagg_plus']
    
    return dfdy


def CalcCoagDeriv(t, vcon, p2):
    """
    CalcCoagDeriv calculates the derivatives in the population balance
    equations.
    """
    
    n_sections = len(vcon)
    
    vcon[vcon < 0] = np.finfo(float).eps
    
    vcon_r = vcon.reshape(-1)  # Ensure vcon is treated as a row vector
    
    vcon_shift = np.hstack((0, vcon_r[:n_sections - 1]))
    
    term1 = vcon_r * p2['b25']
    term1 = vcon_r * term1
    
    term2 = vcon_r @ p2['b1']
    term2 = term2 * vcon_shift
    
    term3 = p2['linear'] @ vcon
    
    # Commented out MATLAB code for term4 is replaced by directly using term3 in Python
    # dvdt calculation includes the sum of term1, term2, and term3 without term4
    
    dvdt = (term1 + term2) + term3
    
    c3 = 0.2
    c4 = 1.45
    
    for isec in range(1, n_sections - 1):  # MATLAB's 2:n_sections-1 becomes 1:n_sections-2 in Python due to zero-indexing
        dvdt[isec] -= c3 * c4 ** (isec + 1) * (vcon[isec] - c4 * vcon[isec + 1])
    
    return dvdt



def CalcRates(vcon, p, p2):
    """
    CalcRates calculates the derivatives in the population balance equations.
    
    Parameters:
    - vcon: Vector of concentrations.
    - p: Parameters structure.
    - p2: Additional parameters structure.
    
    Returns:
    - term1, term2, term3, term4, term5: Calculated terms.
    """
    
    n_sections = len(vcon)
    vcon[vcon < 0] = np.finfo(float).eps
    
    vcon_r = vcon.reshape(-1)  # Ensure vcon is treated as a row vector
    
    vcon_shift = np.hstack((0, vcon_r[:n_sections - 1]))
    
    term1 = vcon_r * p2['b25']
    term1 = vcon_r * term1
    
    term2 = vcon_r @ p2['b1']
    term2 = term2 * vcon_shift
    
    term3 = p2['linear'] @ vcon
    
    # Initializing term4 and term5 as zeros
    term4 = np.zeros(n_sections)
    term5 = np.zeros(n_sections)
    
    c3 = 0.2
    c4 = 1.45
    
    for isec in range(1, n_sections - 1):  # Adjust for Python's zero-based indexing
        term4[isec] = c3 * c4 ** (isec + 1) * vcon[isec]
        term5[isec] = c3 * c4 ** (isec + 2) * vcon[isec + 1]
        # The dvdt calculation is omitted as it's not required for returning terms
    
    return term1, term2, term3, term4, term5


def SectionalMassBalance(spec, p2):
    n_times, n_sections = spec.shape

    # Initialize arrays for coagulation gains and losses
    coag_gains = np.zeros((n_times, n_sections))
    coag_losses = np.zeros((n_times, n_sections))

    # Loop over time to calculate coagulation gains and losses
    for i_time in range(n_times):
        vcon_r = spec[i_time, :]
        vcon_shift = np.hstack((0, vcon_r[:-1]))
        
        term1 = vcon_r @ p2['b2']
        term1 = vcon_r * term1

        term2 = vcon_r @ p2['b1']
        term2 = term2 * vcon_shift
        
        coag_gains[i_time, :] = term1 + term2
        
        term3 = vcon_r @ (p2['b3'] + p2['b4'] + p2['b5'])
        term3 = vcon_r * term3

        coag_losses[i_time, :] = term3

    # Sinking losses
    sinking = np.diag(p2['sink_loss']).reshape(1, -1)
    sink_losses = sinking * spec

    # Growth gains and losses
    g_gain = np.diag(p2['growth'], -1)
    g_gain = np.insert(g_gain, 0, p2['growth'][0, 0])
    growth_gain = g_gain.reshape(1, -1) * spec

    g_loss = np.diag(p2['growth'])
    g_loss[0] = 0.0
    growth_loss = g_loss.reshape(1, -1) * spec

    gains = {'coag': coag_gains, 'growth': growth_gain}
    losses = {'coag': coag_losses, 'growth': growth_loss, 'settl': sink_losses}

    return gains, losses


def TotalMassBalance(spec, p2):
    n_times, n_sections = spec.shape

    # Sinking losses
    sinking = np.diag(p2['sink_loss']).reshape(1, -1)
    sink_losses = sinking * spec
    total_sink_losses = np.sum(sink_losses, axis=1)

    # Growth inputs
    net_growth = np.zeros(n_times)

    for i_time in range(n_times):
        v = spec[i_time, :].reshape(-1, 1)
        g1 = p2['growth'] @ v
        net_growth[i_time] = np.sum(g1)

    # Coagulation losses
    coag_losses = np.zeros(n_times)

    for i_time in range(n_times):
        v = spec[i_time, :]
        v_r = v.reshape(-1, 1)

        term1 = p2['b4'][-1, -1] * v[-1] ** 2
        
        term2 = (p2['b5'][-1, :] * v) * v[-1]
        
        term3 = (p2['b2'][:, -1].reshape(-1, 1) * v_r) * v[-1]
        
        term4 = term2 - term3.sum()
        
        term5 = (p2['b3'][:, -1].reshape(-1, 1) * v_r) * v[-1]
        term5 = term5.sum()
        
        coag_losses[i_time] = term1 + term4 + term5

    total_gains = {'growth': net_growth}
    total_losses = {'sett': total_sink_losses, 'coag': coag_losses}

    return total_gains, total_losses



def CoagOutput(p, p2, t_out, spec):
    # Preliminary setup and calculations (simplified example)
    n_times, n_sections = spec.shape
    vcon = np.clip(spec, np.finfo(float).eps, None)  # Ensure non-negative values
    
    # Assume SectionalMassBalance and TotalMassBalance functions are defined elsewhere
    sec_gains, sec_losses = SectionalMassBalance(spec, p2)
    total_gains, total_losses = TotalMassBalance(spec, p2)
    
    # Example calculation (simplified)
    r_i = p['amfrac'] * p['av_vol'] ** p['bmfrac']
    r_v = (0.75 / np.pi * p['av_vol']) ** (1.0 / 3.0)
    
    # Example plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_out, total_gains / total_losses, label='Gains/Losses')
    plt.xlabel('Time [d]')
    plt.ylabel('Gains/Losses')
    plt.title('Total System Mass Balance')
    plt.legend()
    plt.show()

    # Additional plots and calculations would follow here, following the structure above

    return True  # Indicate successful output








if __name__ == "__main__":
    # Set up and get user options
    p, opt = SetUpCoag()

    # Calculate the sectionally integrated coagulation kernels
    print('Calculating kernels')
    b_brown = CalcBetas(p)
    b_brown.b1 = b_brown.b1 * p.conBr * p.day_to_sec
    b_brown.b2 = b_brown.b2 * p.conBr * p.day_to_sec
    b_brown.b3 = b_brown.b3 * p.conBr * p.day_to_sec
    b_brown.b4 = b_brown.b4 * p.conBr * p.day_to_sec
    b_brown.b5 = b_brown.b5 * p.conBr * p.day_to_sec

    p.kernel = 'KernelCurSh'
    b_shear = CalcBetas(p)
    b_shear.b1 = b_shear.b1 * p.gamma * p.day_to_sec
    b_shear.b2 = b_shear.b2 * p.gamma * p.day_to_sec
    b_shear.b3 = b_shear.b3 * p.gamma * p.day_to_sec
    b_shear.b4 = b_shear.b4 * p.gamma * p.day_to_sec
    b_shear.b5 = b_shear.b5 * p.gamma * p.day_to_sec
    b_shear.b25 = b_shear.b25 * p.gamma * p.day_to_sec

    p.kernel = 'KernelCurDS'
    b_ds = CalcBetas(p)
    b_ds.b1 = b_ds.b1 * p.setcon * p.day_to_sec
    b_ds.b2 = b_ds.b2 * p.setcon * p.day_to_sec
    b_ds.b3 = b_ds.b3 * p.setcon * p.day_to_sec
    b_ds.b4 = b_ds.b4 * p.setcon * p.day_to_sec
    b_ds.b5 = b_ds.b5 * p.setcon * p.day_to_sec
    b_ds.b25 = b_ds.b25 * p.setcon * p.day_to_sec

    # Pack up the betas and store them in a new structure that will get passed
    # to the derivative and jacobian calculation routines
    p2.b1 = b_brown.b1 + b_shear.b1 + b_ds.b1
    p2.b2 = b_brown.b2 + b_shear.b2 + b_ds.b2
    p2.b3 = b_brown.b3 + b_shear.b3 + b_ds.b3
    p2.b4 = b_brown.b4 + b_shear.b4 + b_ds.b4
    p2.b5 = b_brown.b5 + b_shear.b5 + b_ds.b5

    p2.b25 = p2.b2 - p2.b3 - p2.b4 - p2.b5

    # Calculate the linear terms in the population balance equation
    p2.growth = CalcGrowth(p)
    p2.sink_loss = CalcSinkingLoss(p)
    p2.linear = p2.growth - p2.sink_loss

    # Calculate disaggregation terms
    p2.disagg_minus = p.c3 * np.diag(p.c4 ** (np.arange(1, p.n_sections + 1)))
    p2.disagg_plus = p.c3 * np.diag(p.c4 ** (np.arange(2, p.n_sections + 1)), -1)

    # Initial Size Spectrum
    spec_init = CalcInitialSpec(p, p2)

    # Integrate Coagulation Equations
    print('Solving ODEs')
    calcomp = np.arange(1, p.n_sections + 1)
    abs_tol = 1.0e-18  # Absolute Tolerance baseline
    rel_tol = 3.0e-14  # Relative tolerance
    at = abs_tol * 1.5 ** (-(calcomp - 1))
    t_span = np.arange(p.t_init, p.t_final, p.delta_t)
    ode_options = {'reltol': rel_tol, 'refine': 0, 'abstol': at, 'jac': CalcCoagJac}
    t_out, y = ode15s(CalcCoagDeriv, t_span, spec_init, args=(p2,), **ode_options)

    
    # Output
    t1_out = np.zeros((p.n_sections, len(t_out)))
    t2_out = np.zeros((p.n_sections, len(t_out)))
    t3_out = np.zeros((p.n_sections, len(t_out)))
    t4_out = np.zeros((p.n_sections, len(t_out)))
    t5_out = np.zeros((p.n_sections, len(t_out)))

    for itime in range(len(t_out)):
        yvalues = y[itime, :].reshape(-1, 1)
        t1, t2, t3, t4, t5 = CalcRates(yvalues, p, p2)
        t1_out[:, itime] = t1.flatten()
        t2_out[:, itime] = t2.flatten()
        t3_out[:, itime] = t3.flatten()
        t4_out[:, itime] = t4.flatten()
        t5_out[:, itime] = t5.flatten()

    outflag = CoagOutput(p, p2, t_out, y)
