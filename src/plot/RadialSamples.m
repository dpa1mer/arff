function [points, angles] = RadialSamples(base, R, phase)

r = repelem(1:R, base * (1:R)).';
z = exp(1i * phase) .* r .* cumprod(exp(2i * pi ./ (base .* r)));
points = [real(z) imag(z) zeros(size(z))];
angles = angle(z);

end

