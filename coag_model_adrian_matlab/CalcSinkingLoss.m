function sink_loss = CalcSinkingLoss(p)
%
% CalcExport calculates specific loss out of the layer because of particles
% sinking


fractal_radius   = p.amfrac*p.av_vol.^p.bmfrac;
conserved_radius = (0.75/pi*p.av_vol).^(1.0/3.0);

% print fractal

settling_vely = SettlingVelocity(fractal_radius, conserved_radius, p.setcon);

sink_loss = diag(settling_vely)/100*p.day_to_sec/p.dz;


