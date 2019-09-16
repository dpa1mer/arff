%modified from jsolomon\octahedral_frames\code\harmonics\plotY4.m
function PlotRealY4(realCoeffs, shift, scale, center, nTicks)

if nargin < 2
    shift = 1;
end

if nargin < 3
    scale = 1;
end

if nargin < 4
    center = [0 0 0];
end

if nargin < 5
    nTicks = 100;
end

sphx = @(u, v) sin(u) .* cos(v);
sphy = @(u, v) sin(u) .* sin(v);
sphz = @(u, v) cos(u);

[u, v] = meshgrid(linspace(0, pi, nTicks), linspace(0, 2*pi, nTicks));
rData = RealY4Basis(realCoeffs, u, v);
rDataScaledShifted = scale * rData + shift;
xData = center(1) + rDataScaledShifted .* sphx(u, v);
yData = center(2) + rDataScaledShifted .* sphy(u, v);
zData = center(3) + rDataScaledShifted .* sphz(u, v);

surface(xData, yData, zData, rData, 'EdgeColor', 'none');
shading interp;
% colormap winter;
% whitebg('black');
end