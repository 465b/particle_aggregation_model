function [total_gains, total_losses] = TotalMassBalance(spec, p2)
%
% TotalMassBalance does a total mass balance on the system
%

[n_times, n_sections] = size(spec);

% First calculate the total sinking losses from the system

sinking = diag(p2.sink_loss);
sinking = sinking';
sinking = sinking(ones(n_times,1),:);

sink_losses = sinking.*spec;

total_sink_losses = sum(sink_losses, 2);

% Now calculate growth inputs - recall growth can occur in multiple
% sections.

net_growth = zeros(n_times,1);

for i_time = 1 : n_times
    
    v = spec(i_time,:)';
    g1 = p2.growth*v;
    net_growth(i_time) = sum(g1);
    
end

% Coagulation losses from the system

coag_losses = zeros(n_times, 1);


for i_time = 1 : n_times
    
    v   = spec(i_time,:);
    v_r = v';

    term1 = p2.b4(n_sections,n_sections)*v(n_sections)*v(n_sections);
        
    term2 = (p2.b5(n_sections,:).*v)*v(n_sections);
    term2 = term2';
    
    term3 = (p2.b2(:,n_sections).*v')*v(n_sections);
    
    term4 = term2 - term3;
    
    term4 = sum(term4');

    term5 = (p2.b3(:,n_sections).*v')*v(n_sections);
    term5 = sum(term5');
    
    
    coag_losses(i_time) = term1 + term4 + term5;
end
keyboard

total_gains.growth = net_growth;
total_losses.sett  = total_sink_losses;
total_losses.coag  = coag_losses;


