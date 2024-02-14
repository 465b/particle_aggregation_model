function spec_init = CalcInitialSpec(p, p2)
%
% CalcInitialSpec calculates the initial spectrum. At the present, this is
% based on George's code and calculates a spectrum with equal volume in
% each section
%

spec_init = p.av_vol(1)*ones(p.n_sections, 1);

tfactor = 10.^(0 : p.n_sections-1)';

spec_init = max(spec_init ./ tfactor, 1.0e-30);

spec_init = spec_init * p.num_1;


% % Try a steady state solution as G does
% 
% Vcon        = ones(p.n_sections,1)*p.av_vol(1);
% Vcon(2:end) = Vcon(2:end)/10;        % decrease the initial concentration of others
% Vcon        = p.num_1*Vcon;          % use the initialization data
% 
% 
% Vtry = fsolve(@(x) finitial(x, Vcon(1), p2), Vcon(2:end)*1e7);
% 
% Vcon(2:end)=Vtry/1e7;
% 
% spec_init = Vcon;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function F = finitial(x, s1conc, p)
%
% Function to optimize to find the steady state spectrum
%


xx=[s1conc;x/1e7];      % have to de scale, add first section conc
FF=CalcCoagDeriv(0,xx,p);
F=FF(2:end)*1e7;        % rescale

