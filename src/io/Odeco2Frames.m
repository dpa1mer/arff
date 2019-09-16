function frames = Odeco2Frames(odecoCoeffs)
n = size(odecoCoeffs, 2);

% syms x y z;
% degrees = MonomialDegrees(4);
% coeffs = (1./prod(factorial(degrees)));
% odecoDx = jacobian(coeffs .* prod([x; y; z].^degrees), x).';
% odecoDy = jacobian(coeffs .* prod([x; y; z].^degrees), y).';
% odecoDz = jacobian(coeffs .* prod([x; y; z].^degrees), z).';
% odecoDx = matlabFunction(odecoDx, 'Vars', {x, y, z});
% odecoDy = matlabFunction(odecoDy, 'Vars', {x, y, z});
% odecoDz = matlabFunction(odecoDz, 'Vars', {x, y, z});

v1 = randn(n, 3);
v2 = randn(n, 3);
delta = 1;
while delta > eps^0.9 * sqrt(n)
    v1Old = v1;

    w1 = OdecoTensorGradient(odecoCoeffs, v1);
    w2 = OdecoTensorGradient(odecoCoeffs, v2);

    v1 = dot(v1, w1, 2) .* w1;
    v1 = v1 ./ vecnorm(v1, 2, 2);
    w2 = w2 - dot(w2, v1, 2) .* v1;
    v2 = dot(v2, w2, 2) .* w2;
    v2 = v2 ./ vecnorm(v2, 2, 2);
    
    delta = norm(v1 - v1Old, 'fro');
end

v3 = cross(v1, v2, 2);
w3 = OdecoTensorGradient(odecoCoeffs, v3);

frames = [permute(w1, [2 3 1]) permute(w2, [2 3 1]) permute(w3, [2 3 1])] / 4;

end
