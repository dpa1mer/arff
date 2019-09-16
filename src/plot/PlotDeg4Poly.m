% Plot homogeneous polynomials of degree 4 over the sphere
function PlotDeg4Poly(coeffs, shift, scale, center, nTicks)

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

degree = MonomialDegrees(4);
scaledCoeffs = (1./prod(factorial(degree)))' .* coeffs;

sphx = @(u, v) sin(u) .* cos(v);
sphy = @(u, v) sin(u) .* sin(v);
sphz = @(u, v) cos(u);

[u, v] = meshgrid(linspace(0, pi, nTicks), linspace(0, 2*pi, nTicks));
xData = sphx(u, v);
yData = sphy(u, v);
zData = sphz(u, v);

shape = size(xData);

rData = reshape((xData(:).^degree(1,:) .* yData(:).^degree(2,:) .* zData(:).^degree(3,:)) * scaledCoeffs, shape);
rDataScaledShifted = shift + scale * rData;
xData = center(1) + xData .* rDataScaledShifted;
yData = center(2) + yData .* rDataScaledShifted;
zData = center(3) + zData .* rDataScaledShifted;

surface(xData, yData, zData, rData, 'EdgeColor', 'none');
% shading interp;
% colormap winter;
% whitebg('black');
end