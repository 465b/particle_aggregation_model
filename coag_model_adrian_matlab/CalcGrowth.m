function growth = CalcGrowth(p)
%
% CalcGrowth calculates the matrix of terms that represent algal cell
% growth in the population balance equation
%
%

growth_loss = zeros(p.n_sections, 1);
growth_gain = zeros(p.n_sections-1, 1);

if p.gro_sec > 0
    
    growth_loss(p.gro_sec : p.n_sections -1) = -1;
    growth_gain(p.gro_sec : end) = 2;
    
end

growth = diag(growth_loss) + diag(growth_gain, -1);
growth(1,1) = 1;

growth = p.growth*growth;
