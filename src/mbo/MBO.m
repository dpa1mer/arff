function [q, q0, info] = MBO(meshData, fiber, q0, tauMult, tauExponent, saveIterates)
%% Diffusion-generated algorithm for variety-valued maps.

nv = meshData.nv;
bdryIdx = meshData.bdryIdx;
intIdx = meshData.intIdx;
bdryNormals = meshData.bdryNormals;
L = meshData.L;
M = meshData.M;

%% Constraint setup

[bdryFixed, bdryBasis] = fiber.bdryBasis(bdryNormals);
qFixed = zeros(fiber.dim, nv);
qFixed(:, bdryIdx) = bdryFixed;


%% Initial Value

if nargin < 3 || isempty(q0)
    q0 = fiber.rand(nv, bdryIdx, bdryNormals);
end


%% Optimization Schedule

if nargin < 4 || isempty(tauMult)
    tauMult = 1;
end

tau0 = tauMult / meshData.lambda1L;

if nargin < 5
    tauExponent = 0;
end

%% MBO

if nargin < 6
    saveIterates = false;
end

q = q0;
warmstart = [];
qProj = zeros(size(q));

k = 1;
info(k).cost = 0.5 * sum(dot(q.', L * q.'));
info(k).costdelta = info(k).cost;
info(k).gradnorm = norm(L * q.', 'fro');
info(k).time = 0;
info(k).tau = tau0;
if saveIterates
    info(k).q = q;
end

tic;
for k = 2:1000
    info(k).tau = tau0 / (k - 1)^tauExponent;
    A = M + info(k).tau * L;
    
    % Relax
    [qDiffused, iter, warmstart] = AlignmentConstrainedLinearSolve( ...
        A, (M * q.').', intIdx, ...
        bdryIdx, bdryBasis, qFixed, warmstart);
    
    % Project
    qDiffBdry = multiprod(multitransp(bdryBasis), qDiffused(:, bdryIdx) - bdryFixed, [1 2], 1);
    qProjBdry = fiber.projAligned(qDiffBdry);
    qProj(:, bdryIdx) = multiprod(bdryBasis, qProjBdry, [1 2], 1) + bdryFixed;
    qProj(:, intIdx) = fiber.proj(qDiffused(:, intIdx));
    
    % Compute Statistics
    info(k).delta = sqrt(sum(dot((qProj - q).', M * (qProj - q).')));
    q = qProj;
    info(k).cost = 0.5 * sum(dot(q.', L * q.'));
    info(k).costdelta = info(k - 1).cost - info(k).cost;
    info(k).gradnorm = norm(L * q.', 'fro');
    info(k).time = toc;
    if saveIterates
        info(k).q = q;
    end
    fprintf("t = %3.3gs, cost = %3.6g, delta = %3.3g, inner iters = %d\n", ...
            info(k).time, info(k).cost, info(k).delta, iter);
    
    if abs(info(k).costdelta / info(k).cost) < 1e-5 || info(k).delta / sqrt(sum(dot(q.', M * q.'))) < 1e-5
        break;
    end
end

end

