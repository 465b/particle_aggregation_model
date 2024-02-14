function v = SettlingVelocity(r, rcons, sett_const)
%
% Settling Velocity calculates the settling velocities of particles of
% given sizes.
%
% USAGE:
%   v = SettlingVelocity(r, rcons, sett_const)
%
%   v = particle settling velocities [cm s^{-1}]
%   r = particle radii [cm]
%   rcons = radii of particle conserved volumes [cm]
%   sett_const = (2g/9eta)*(delta_rho/rho_fluid)
%
% HISTORY:
%   23-04-09: First Cut
%
% Adrian Burd, University of Georgia, 2009

v = sett_const * rcons.*rcons.*rcons./r;

