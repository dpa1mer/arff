function [worstEig2, eigRatio, q0, q, Q] = OdecoExactnessTest(n)

clebschGordan(:,:,1)=[(1/2).*pi.^(-1/2),0,0,0,0,0;0,(1/2).*pi.^(-1/2),0,0,0,0;0,0,(1/2).*pi.^(-1/2), ...
  0,0,0;0,0,0,(1/2).*pi.^(-1/2),0,0;0,0,0,0,(1/2).*pi.^(-1/2),0;0,0,0,0,0,(1/2).* ...
  pi.^(-1/2)];
clebschGordan(:,:,2)=[0,(1/2).*pi.^(-1/2),0,0,0,0;(1/2).*pi.^(-1/2),0,0,(-1/7).*(5.*pi.^(-1)).^(1/2), ...
  0,0;0,0,0,0,(1/14).*(15.*pi.^(-1)).^(1/2),0;0,(-1/7).*(5.*pi.^(-1)).^(1/2),0,0, ...
  0,0;0,0,(1/14).*(15.*pi.^(-1)).^(1/2),0,0,0;0,0,0,0,0,0];
clebschGordan(:,:,3)=[0,0,(1/2).*pi.^(-1/2),0,0,0;0,0,0,0,(1/14).*(15.*pi.^(-1)).^(1/2),0;(1/2).* ...
  pi.^(-1/2),0,0,(1/14).*(5.*pi.^(-1)).^(1/2),0,(-1/14).*(15.*pi.^(-1)).^(1/2);0, ...
  0,(1/14).*(5.*pi.^(-1)).^(1/2),0,0,0;0,(1/14).*(15.*pi.^(-1)).^(1/2),0,0,0,0;0, ...
  0,(-1/14).*(15.*pi.^(-1)).^(1/2),0,0,0];
clebschGordan(:,:,4)=[0,0,0,(1/2).*pi.^(-1/2),0,0;0,(-1/7).*(5.*pi.^(-1)).^(1/2),0,0,0,0;0,0,(1/14).* ...
  (5.*pi.^(-1)).^(1/2),0,0,0;(1/2).*pi.^(-1/2),0,0,(1/7).*(5.*pi.^(-1)).^(1/2),0, ...
  0;0,0,0,0,(1/14).*(5.*pi.^(-1)).^(1/2),0;0,0,0,0,0,(-1/7).*(5.*pi.^(-1)).^(1/2) ...
  ];
clebschGordan(:,:,5)=[0,0,0,0,(1/2).*pi.^(-1/2),0;0,0,(1/14).*(15.*pi.^(-1)).^(1/2),0,0,0;0,(1/14).*( ...
  15.*pi.^(-1)).^(1/2),0,0,0,0;0,0,0,0,(1/14).*(5.*pi.^(-1)).^(1/2),0;(1/2).*pi.^( ...
  -1/2),0,0,(1/14).*(5.*pi.^(-1)).^(1/2),0,(1/14).*(15.*pi.^(-1)).^(1/2);0,0,0,0,( ...
  1/14).*(15.*pi.^(-1)).^(1/2),0];
clebschGordan(:,:,6)=[0,0,0,0,0,(1/2).*pi.^(-1/2);0,0,0,0,0,0;0,0,(-1/14).*(15.*pi.^(-1)).^(1/2),0,0, ...
  0;0,0,0,0,0,(-1/7).*(5.*pi.^(-1)).^(1/2);0,0,0,0,(1/14).*(15.*pi.^(-1)).^(1/2), ...
  0;(1/2).*pi.^(-1/2),0,0,(-1/7).*(5.*pi.^(-1)).^(1/2),0,0];
clebschGordan(:,:,7)=[0,0,0,0,0,0;0,0,0,0,0,(1/2).*((5/7).*pi.^(-1)).^(1/2);0,0,0,0,0,0;0,0,0,0,0,0; ...
  0,0,0,0,0,0;0,(1/2).*((5/7).*pi.^(-1)).^(1/2),0,0,0,0];
clebschGordan(:,:,8)=[0,0,0,0,0,0;0,0,0,0,(1/2).*((5/14).*pi.^(-1)).^(1/2),0;0,0,0,0,0,(1/2).*((5/14) ...
  .*pi.^(-1)).^(1/2);0,0,0,0,0,0;0,(1/2).*((5/14).*pi.^(-1)).^(1/2),0,0,0,0;0,0,( ...
  1/2).*((5/14).*pi.^(-1)).^(1/2),0,0,0];
clebschGordan(:,:,9)=[0,0,0,0,0,0;0,0,0,(1/14).*(15.*pi.^(-1)).^(1/2),0,0;0,0,0,0,(1/7).*(5.*pi.^(-1) ...
  ).^(1/2),0;0,(1/14).*(15.*pi.^(-1)).^(1/2),0,0,0,0;0,0,(1/7).*(5.*pi.^(-1)).^( ...
  1/2),0,0,0;0,0,0,0,0,0];
clebschGordan(:,:,10)=[0,0,0,0,0,0;0,0,0,0,(-1/14).*((5/2).*pi.^(-1)).^(1/2),0;0,0,0,(1/7).*((15/2).* ...
  pi.^(-1)).^(1/2),0,(1/14).*((5/2).*pi.^(-1)).^(1/2);0,0,(1/7).*((15/2).*pi.^(-1) ...
  ).^(1/2),0,0,0;0,(-1/14).*((5/2).*pi.^(-1)).^(1/2),0,0,0,0;0,0,(1/14).*((5/2).* ...
  pi.^(-1)).^(1/2),0,0,0];
clebschGordan(:,:,11)=[0,0,0,0,0,0;0,(1/14).*pi.^(-1/2),0,0,0,0;0,0,(-2/7).*pi.^(-1/2),0,0,0;0,0,0,( ...
  3/7).*pi.^(-1/2),0,0;0,0,0,0,(-2/7).*pi.^(-1/2),0;0,0,0,0,0,(1/14).*pi.^(-1/2)]; ...
  
clebschGordan(:,:,12)=[0,0,0,0,0,0;0,0,(-1/14).*((5/2).*pi.^(-1)).^(1/2),0,0,0;0,(-1/14).*((5/2).* ...
  pi.^(-1)).^(1/2),0,0,0,0;0,0,0,0,(1/7).*((15/2).*pi.^(-1)).^(1/2),0;0,0,0,(1/7) ...
  .*((15/2).*pi.^(-1)).^(1/2),0,(-1/14).*((5/2).*pi.^(-1)).^(1/2);0,0,0,0,(-1/14) ...
  .*((5/2).*pi.^(-1)).^(1/2),0];
clebschGordan(:,:,13)=[0,0,0,0,0,0;0,0,0,0,0,0;0,0,(-1/7).*(5.*pi.^(-1)).^(1/2),0,0,0;0,0,0,0,0,(1/14) ...
  .*(15.*pi.^(-1)).^(1/2);0,0,0,0,(1/7).*(5.*pi.^(-1)).^(1/2),0;0,0,0,(1/14).*( ...
  15.*pi.^(-1)).^(1/2),0,0];
clebschGordan(:,:,14)=[0,0,0,0,0,0;0,0,(-1/2).*((5/14).*pi.^(-1)).^(1/2),0,0,0;0,(-1/2).*((5/14).* ...
  pi.^(-1)).^(1/2),0,0,0,0;0,0,0,0,0,0;0,0,0,0,0,(1/2).*((5/14).*pi.^(-1)).^(1/2); ...
  0,0,0,0,(1/2).*((5/14).*pi.^(-1)).^(1/2),0];
clebschGordan(:,:,15)=[0,0,0,0,0,0;0,(-1/2).*((5/7).*pi.^(-1)).^(1/2),0,0,0,0;0,0,0,0,0,0;0,0,0,0,0,0; ...
  0,0,0,0,0,0;0,0,0,0,0,(1/2).*((5/7).*pi.^(-1)).^(1/2)];

%% Generate random positive polynomials

randM = randn(6, 6, 1, n);
randPSD = multiprod(multitransp(randM), randM);
q0 = squeeze(sum(sum(randPSD .* clebschGordan)));
q0 = q0 ./ vecnorm(q0, 2, 1);

%% Test exactness

odecoMat = LoadOdecoMatsSph;
odecoMat = cell2mat(permute(odecoMat, [2 3 1]));
nConstr = size(odecoMat, 3);
odecoMat = [zeros(1, 16, nConstr); zeros(15, 1, nConstr), odecoMat];
odecoMat = cat(3, odecoMat, blkdiag(1, zeros(15)));
odecoMat = reshape(odecoMat, [256, nConstr + 1])';
odecoRhs = [zeros(nConstr, 1); 1];

[q, Q] = MultiSdp(q0, odecoMat, odecoRhs);

Q = reshape(Q, 16, 16, n);
firstEig = zeros(n, 1);
secondEig = zeros(n, 1);
for i = 1:n
    ev = eig(Q(:, :, i));
    firstEig(i) = ev(16);
    secondEig(i) = ev(15);
end

eigRatio = secondEig ./ firstEig;
figure; histogram(log10(eigRatio), 'LineStyle', 'none');
worstEig2 = max(secondEig);

end