function grad = OdecoTensorGradient(odecoCoeffs, v)
n = size(v, 1);
assert(n == size(odecoCoeffs, 2));

x = permute(v(:, 1), [3 2 1]);
y = permute(v(:, 2), [3 2 1]);
z = permute(v(:, 3), [3 2 1]);

Jacobian = ...
 [zeros(1,1,n),(1/6).*z.^3,(1/2).*x.*z.^2,(1/2).*x.^2.*z,(1/6).*x.^3,zeros(1,1,n),(1/2).*y.*z.^2, ...
  x.*y.*z,(1/2).*x.^2.*y,zeros(1,1,n),(1/2).*y.^2.*z,(1/2).*x.*y.^2,zeros(1,1,n),(1/6).*y.^3,zeros(1,1,n); ...
  zeros(1,5,n),(1/6).*z.^3,(1/2).*x.*z.^2,(1/2).*x.^2.*z,(1/6).*x.^3, ...
  (1/2).*y.*z.^2,x.*y.*z,(1/2).*x.^2.*y,(1/2).*y.^2.*z,(1/2).*x.*y.^2,(1/6).*y.^3; ...
  (1/6).*z.^3,(1/2).*x.*z.^2,(1/2).*x.^2.*z,(1/6).*x.^3,zeros(1,1,n),(1/2).*y.*z.^2,x.*y.*z, ...
  (1/2).*x.^2.*y,zeros(1,1,n),(1/2).*y.^2.*z,(1/2).*x.*y.^2,zeros(1,1,n),(1/6).*y.^3,zeros(1,2,n)];

grad = squeeze(multiprod(Jacobian, odecoCoeffs, 2, 1)).';

end