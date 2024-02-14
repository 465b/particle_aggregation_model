function betas = CalcBetas_vs(p)
%
% CalcBetas calculate the sectionally integrated coagulation kernel
% coefficients.
%
% The coagulation coefficients are returned in a structure (betas) with
% each each element of the structure being a matrix of size (NxN) with N
% being the requested number of sections. In this case, the columns
% represent those sections that are losing or gaining material through
% coagulation, and the rows represent the section for the other particle
% involved. 
%
% Note: Some of this code is adapted from the Matlab coagulation model
% available from George Jackson's website.
%
% USAGE:
%
% HISTORY:
%   23-04-09: First Cut, add kernel formulations
%
% Adrian Burd, University of Georgia, 2009
%

n_sections = p.n_sections;
mlo = p.v_lower;


% Set up constants and variables. Place constants into a structure.

int_param.amfrac = [];
int_param.bmfrac = [];
int_param.kernel = [];
int_param.constants = [];

beta_init = zeros(n_sections, n_sections);



% betas.b5 = loss from jcol by collisions of jcol & irow > jcol

b5 = beta_init;

for jcol = 1 : (n_sections - 1)
    
    for irow = (jcol + 1) : n_sections
        
        mj_lo = mlo(jcol);
        mj_up = 2.0*mj_lo;
        mi_lo = mlo(irow);
        mi_up = 2.0*mi_lo;
        
        bndry.mi_lo = mi_lo;
        bndry.mi_up = mi_up;
        bndry.mj_lo = mj_lo;
        bndry.mj_up = mj_up;
        bndry.mjj   = [];
        bndry.rjj   = [];
        bndry.rvj   = [];
        
        b5(irow, jcol) = quadl(@(x) integr5a(x, p, bndry), mi_lo, mi_up)/(mi_lo*mj_lo);

    end
end

betas.b5 = b5;

% betas.b4 = loss from jcol by collisions with itself

b4 = beta_init;

for jcol = 1 : n_sections
    
    mj_lo = mlo(jcol);
    mj_up = 2.0*mj_lo;
    mi_lo = mlo(jcol);
    mi_up = 2.0*mi_lo;

    bndry.mi_lo = mi_lo;
    bndry.mi_up = mi_up;
    bndry.mj_lo = mj_lo;
    bndry.mj_up = mj_up;
    bndry.mjj   = [];
    bndry.rjj   = [];
    bndry.rvj   = [];
    
    b4(jcol, jcol) = quadl(@(x) integr4a(x, p, bndry), mi_lo, mi_up)/(mi_lo*mj_lo);

end

betas.b4 = b4;

% betas.b3 = loss from jcol by collisions of jcol & irow < jcol

b3 = beta_init;

for jcol = 2 : n_sections
    
    for irow = 1 : (jcol - 1)
        
    mj_lo = mlo(jcol);
    mj_up = 2.0*mj_lo;
    mi_lo = mlo(irow);
    mi_up = 2.0*mi_lo;

    bndry.mi_lo = mi_lo;
    bndry.mi_up = mi_up;
    bndry.mj_lo = mj_lo;
    bndry.mj_up = mj_up;
    bndry.mjj   = [];
    bndry.rjj   = [];
    bndry.rvj   = [];

    b3(irow, jcol) = quadl(@(x) integr3a(x, p, bndry), mi_lo, mi_up)/(mi_lo*mj_lo);
        
    end
    
end

betas.b3 = b3;

% betas.b2 = gain in jcol by collisions of jcol & irow < jcol

b2 = beta_init;

for jcol = 2 : n_sections
    
    for irow = 1 : (jcol - 1)
        
    mj_lo = mlo(jcol);
    mj_up = 2.0*mj_lo;
    mi_lo = mlo(irow);
    mi_up = 2.0*mi_lo;

    bndry.mi_lo = mi_lo;
    bndry.mi_up = mi_up;
    bndry.mj_lo = mj_lo;
    bndry.mj_up = mj_up;
    bndry.mjj   = [];
    bndry.rjj   = [];
    bndry.rvj   = [];

    
    b2(irow, jcol) = quadl(@(x) integr2a(x, p, bndry), mi_lo, mi_up)/(mi_lo*mj_lo);
    
    end
    
end

betas.b2 = b2;

% betas.b1 = gain in jcol by collisions of (jcol-1) & irow < jcol

b1 = beta_init;

for jcol = 2 : n_sections

    for irow = 1 : jcol - 1
        
    mj_lo = mlo(jcol-1);
    mj_up = 2.0*mj_lo;
    mi_lo = mlo(irow);
    mi_up = 2.0*mi_lo;
 
    bndry.mi_lo = mi_lo;
    bndry.mi_up = mi_up;
    bndry.mj_lo = mj_lo;
    bndry.mj_up = mj_up;
    bndry.mjj   = [];
    bndry.rjj   = [];
    bndry.rvj   = [];
       
    b1(irow, jcol) = quadl(@(x) integr1a(x, p, bndry), mi_lo, mi_up)/(mi_lo*mj_lo);
    
    end
    
end

betas.b1 = b1;

betas.b25 = b2 - b3 - b4 - b5;

% DON'T FORGET TO MULTIPLY BROWNIAN BY 2/3(KT/MU) AND SHEAR BY GAMMA BEFORE
% PASSING BACK TO CALLING ROUTINE


%%%%%%%%%%%%%%%%%%%%%% Last line of CalcBetas %%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Outer & inner integrals for case 5: 
%  mass loss from section i resulting from collisions with particles in 
%  section j, j > i
%
function x = integr5a(mj, param, bndry)
%
%

nj = length(mj); x = 0*mj;
rj = param.amfrac*mj.^param.bmfrac;
rvj =  (0.75/pi*mj).^(1.0/3.0);

for iv = 1 : nj
    bndry.mjj  = mj(iv);
    bndry.rjj  = rj(iv);
    bndry.rvjj = rvj(iv);
    
    x(iv) = quadl(@(y) integr5b(y, param, bndry), bndry.mj_lo, bndry.mj_up);
end

x = x./mj;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yint = integr5b(mi, param, bndry)

ri = param.amfrac*mi.^(param.bmfrac);

ni = length(mi); 
rj = bndry.rjj*ones(1, ni);
mj = bndry.mjj*ones(1, ni);

rvi = (0.75/pi*mi).^(1.0/3.0);
rvj = bndry.rvjj*ones(1,ni);

yint = feval(param.kernel, [ri; rj], [rvi; rvj], param);


%% Outer and inner integrals for case 4: 
%  mass loss from section i resulting from collisions of particles 
%  within section i
%
function x = integr4a(mj, param, bndry)
%
%

nj = length(mj); x = 0*mj;
rj = param.amfrac*mj.^param.bmfrac;
rvj =  (0.75/pi*mj).^(1.0/3.0);

for iv = 1 : nj
    bndry.mjj  = mj(iv);
    bndry.rjj  = rj(iv);
    bndry.rvjj = rvj(iv);
    
    x(iv) = quadl(@(y) integr4b(y, param, bndry), bndry.mj_lo, bndry.mj_up);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yint = integr4b(mi, param, bndry)

ri = param.amfrac*mi.^(param.bmfrac);

ni = length(mi); 
rj = bndry.rjj*ones(1, ni);
mj = bndry.mjj*ones(1, ni);

rvi = (0.75/pi*mi).^(1.0/3.0);
rvj = bndry.rvjj*ones(1,ni);

yint = feval(param.kernel, [ri; rj], [rvi; rvj], param);

yint = (mi + bndry.mjj)./mi./bndry.mjj.*yint;

%% Outer and inner integrals for case 3: 
%  mass loss from section i resulting from collisions of particles 
%  within section j<i
%
function x = integr3a(mj, param, bndry)
%
%

nj = length(mj); x = 0*mj;
rj = param.amfrac*mj.^param.bmfrac;
rvj =  (0.75/pi*mj).^(1.0/3.0);

for iv = 1 : nj
    bndry.mjj  = mj(iv);
    bndry.rjj  = rj(iv);
    bndry.rvjj = rvj(iv);
    
    
    x(iv) = quadl(@(y) integr3b(y, param, bndry), bndry.mj_up-bndry.mjj, bndry.mj_up);
end



x = x./mj;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yint = integr3b(mi, param, bndry)

ri = param.amfrac*mi.^(param.bmfrac);

ni = length(mi); 
rj = bndry.rjj*ones(1, ni);
mj = bndry.mjj*ones(1, ni);

rvi = (0.75/pi*mi).^(1.0/3.0);
rvj = bndry.rvjj*ones(1,ni);

yint = feval(param.kernel, [ri; rj], [rvi; rvj], param);

%% Outer and inner integrals for case 2: 
%  mass gain to section i resulting from collisions of particles 
%  within section j
%
function x = integr2a(mj, param, bndry)
%
%

nj = length(mj); x = 0*mj;
rj = param.amfrac*mj.^param.bmfrac;
rvj =  (0.75/pi*mj).^(1.0/3.0);

for iv = 1 : nj
    bndry.mjj  = mj(iv);
    bndry.rjj  = rj(iv);
    bndry.rvjj = rvj(iv);    
    
    x(iv) = quadl(@(y) integr2b(y, param, bndry), bndry.mj_lo, bndry.mj_up-bndry.mjj);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yint = integr2b(mi, param, bndry)

ri = param.amfrac*mi.^(param.bmfrac);

ni = length(mi); 
rj = bndry.rjj*ones(1, ni);
mj = bndry.mjj*ones(1, ni);

rvi = (0.75/pi*mi).^(1.0/3.0);
rvj = bndry.rvjj*ones(1,ni);

yint = feval(param.kernel, [ri; rj], [rvi; rvj], param);

yint = yint./mi;

%% Outer and inner integrals for case 1: 
%  mass gain to section i+1 resulting from collisions of particles 
%  within section j
%
function x = integr1a(mj, param, bndry)
%
%

nj = length(mj); x = 0*mj;
rj = param.amfrac*mj.^param.bmfrac;
rvj =  (0.75/pi*mj).^(1.0/3.0);

for iv = 1 : nj
    bndry.mjj  = mj(iv);
    bndry.rjj  = rj(iv);
    bndry.rvjj = rvj(iv);    
    mlow       = max([bndry.mj_up - bndry.mjj, bndry.mj_lo]);
   
    x(iv) = quadl(@(y) integr1b(y, param, bndry), mlow, bndry.mj_up);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function yint = integr1b(mi, param, bndry)

ri = param.amfrac*mi.^(param.bmfrac);

ni = length(mi); 
rj = bndry.rjj*ones(1, ni);
mj = bndry.mjj*ones(1, ni);

rvi = (0.75/pi*mi).^(1.0/3.0);
rvj = bndry.rvjj*ones(1,ni);

yint = feval(param.kernel, [ri; rj], [rvi; rvj], param);

yint = yint.*(mi + mj)./mi./mj;


%% Brownian Kernel
%
function b = KernelBrown(r, rcons, param)
%
% USAGE:
%  b = KernelBrown(r, rcons, constants)
%
%  r     = 2xN vector of particle radii [cm]
%  
% rcons and constants are dummy variable passed to make the code
% efficient, so the routine should be called for example as
%
%     b = KernelBrown(r, [], [])


b = param.conBr*(2 + r(1,:)./r(2,:) + r(2,:)./r(1,:));

%% Curvilinear Differential Sedimentation Kernel
%
function b = KernelCurDS(r, rcons, param)
%
% USAGE:
%  b = KernelCurDS(r, rcons, constants)
%
%  r     = column vector of particle radii [cm]
%  rcons = column vector of radii of conserved volume for particles [cm]
%  constants(1) = r_to_rg, factor to convert radius to radius of gyration
%  constants(2) = constant for settling velocity law
%

v = SettlingVelocity(r, rcons, param.setcon);

r_small = min(r)*param.r_to_rg;

b = 0.5*pi*abs(v(1,:) - v(2,:)).*r_small.*r_small;


%% Curvilinear Shear Kernel
%
function b = KernelCurSh(r, rcons, param)
%
% USAGE
%  b = KernelCurSh(r, rcons, r_to_rg)
%
%  r       = column vector of particle radii [cm]
%  r_to_rg = factor to convert radius to radius of gyration
%
%  rcons is a dummy variable

rg = (r(1,:) + r(2,:))*param.r_to_rg;

p  = min(r)./max(r);
p1 = 1.0 + p;
p5 = p1.*p1.*p1.*p1.*p1;

efficiency = 1.0 - (1.0 + 5.0*p + 2.5*p.*p)./p5;

b = sqrt(8.0*pi/15.0) * efficiency.*rg.*rg.*rg;


%% Fractal Differential Sedimentation Kernel
%
function b = KernelFracDS(r, rcons, param)
%
% USAGE:
%   b = KernelFracDS(r, rcons, constants)
%
%  r     = column vector of particle radii [cm]
%  rcons = column vector of radii of conserved volume for particles [cm]
%  constants(1) = r_to_rg, factor to convert radius to radius of gyration
%  constants(2) = constant for settling velocity law

c1 = 0.984;                   % Constant from Li and Logan

rg      = (r(1,:) + r(2,:))*param.r_to_rg;
r_ratio = min(r(1,:)./r(2,:), r(2,:)./r(1,:));

v = SettlingVelocity(r, rcons, param.setcon);

b = pi * abs(v(1,:) - v(2,:)).*rg.*rg;

b = b.*r_ratio.^c1;


%% Fractal Shear Kernel
%
function b = KernelFracSh(r, rcons, param)
%
% USAGE:
%   b = KernelFracSh(r, rcons, r_to_rg)
%
%   r       = column vector of particle radii [cm]
%   r_to_rg = factor to convert radius to radius of gyration
% 
% rcons is a dummy variable

c1      = 0.785;                % Constant from Li and Logan
r_ratio = min(r(1,:)./r(2,:), r(2,:)./r(1,:));
rg      = (r(1,:) + r(2,:))*param.r_to_rg;

b = 1.3*rg.*rg.*rg;

b = b.*r_ratio.^c1;


%% Rectilinear Differential Sedimentation Kernel
%
function b = KernelRectDS(r, rcons, param)
%
% USAGE:
%   b = KernelFracDS(r, rcons, constants)
%
%  r     = column vector of particle radii [cm]
%  rcons = column vector of radii of conserved volume for particles [cm]
%  constants(1) = r_to_rg, factor to convert radiuss to radius of gyration
%  constants(2) = constant for settling velocity law
%   r1, r2         = particle radii [cm]
%   rcons1, rcons2 = radius of conserved volume for particles [cm]
%   r_to_rg        = factor to convert radius to radius of gyration
%   set_const      = constant for settling velocity law
%

rg = (r(1,:) + r(2,:))*param.r_to_rg;
v  = SettlingVelocity(r, rcons, param.setcon);

b = pi * abs(v(1,:) - v(2,:)).*rg.*rg;


%% Rectilinear Shear Kernel
%
function b = KernelRectSh(r, rcons, param)
%
% USAGE:
%   b = KernelFracSh(r, rcons, r_to_rg)
%
%   r       = column vector of particle radii [cm]
%   r_to_rg = factor to convert radius to radius of gyration
% 
% rcons is a dummy variable

rg = (r(1,:) + r(2,:))*param.r_to_rg;
b  = 1.3*rg.*rg.*rg;








