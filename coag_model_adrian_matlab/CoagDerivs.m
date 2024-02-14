function dydt = CoagDerivs(vcon, p2)

n_sections = length(vcon);

dydt = zeros(n_sections, 4);

vcon_r = vcon';      % vcon passed as column vector - make a row

vcon_shift = [0 vcon_r(1:n_sections-1)];

term1 = vcon_r * p2.b25;
term1 = vcon_r .* term1;

term2 = vcon_r * p2.b1;
term2 = term2 .* vcon_shift;

term3 = p2.linear * vcon;

term4a = - diag(p2.disagg_minus).*vcon;
term4b = [diag(p2.disagg_plus,-1);0].*[vcon(2:end);0];

dydt(:,1) = term1';
dydt(:,2) = term2';
dydt(:,3) = term3;
dydt(:,4) = term4a;
dydt(:,5) = term4b;

