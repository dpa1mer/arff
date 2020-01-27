function [q, qInit, info] = Ray(meshData, qInit, useGeometricLaplacian, timeInitialization)

ni = length(meshData.intIdx);
nb = length(meshData.bdryIdx);

if nargin >= 3 && useGeometricLaplacian
    L = GeometricPrimalLM(meshData.verts, meshData.tets);
else
    E = edges(meshData.tetra);
    edgeGraph = graph(E(:, 1), E(:, 2));
    L = laplacian(edgeGraph);
end

if nargin < 4
    timeInitialization = true;
end

[Lx, Ly, Lz, YZ] = LoadSO3Generators_Y4;
XZ = expm(-(pi/2) * Ly);

B = OctaAlignMat(meshData.bdryNormals);

q0 = repmat(OctaZAligned(0), 1, meshData.nv);

%% Initialization
if timeInitialization
    tic;
end
if nargin >= 2 && ~isempty(qInit)
    fInit = Coeff2Frames(qInit);
else
    [qInit, fInit] = RayInit(meshData);
end
eulInitInt = fliplr(rotm2eul(fInit(:, :, meshData.intIdx), 'ZYX'));
qInitBdry = squeeze(batchop('mult', B, permute(qInit(:, meshData.bdryIdx), [1 3 2])));
% Our convention is apparently clockwise, so negate this angle:
eulInitBdry = -0.25 * atan2(-qInitBdry(1, :), qInitBdry(9, :)).';
eulInit = [eulInitBdry; eulInitInt(:)];

%% L-BFGS
problem.objective = @rayObj;
problem.solver = 'fmincon';
problem.x0 = eulInit;
problem.options = optimoptions(@fmincon, 'SpecifyObjectiveGradient', true, ...
                                         'HessianApproximation', 'lbfgs', ...
                                         'CheckGradients', false, ...
                                         'FiniteDifferenceType', 'central', ...
                                         'MaxIterations', 5000, ...
                                         'MaxFunctionEvaluations', 5000, ...
                                         'Display', 'iter', ...
                                         'OptimalityTolerance', 0, ... % Use L2 instead
                                         'OutputFcn', @saveInfo);
if ~timeInitialization
    tic;
end
eul = fmincon(problem);
[eulBdry, eulInt] = splitBdryInt(eul);
[RzRyRxq0, BRbq0] = q0RotatedBy(eulBdry, eulInt);
q(:, meshData.bdryIdx) = BRbq0;
q(:, meshData.intIdx) = RzRyRxq0;

%% Functions defining the objective and gradient
function [obj, grad] = rayObj(eul)
    [eulBdry, eulInt] = splitBdryInt(eul);
    [RzRyRxq0, BRbq0, Rbq0, RyRxq0, Rxq0] = q0RotatedBy(eulBdry, eulInt);
    
    qi(:, meshData.bdryIdx) = BRbq0;
    qi(:, meshData.intIdx) = RzRyRxq0;
    egrad = L * qi.';
    obj = 0.5 * sum(dot(qi.', egrad));
    
    if nargout > 1
        Dzq = permute(Lz * RzRyRxq0, [1 3 2]);
        Dyq = permute(rotateZ(eulInt(:, 3), Ly * RyRxq0), [1 3 2]);
        Dxq = permute(rotateZ(eulInt(:, 3), rotateY(eulInt(:, 2), Lx * Rxq0)), [1 3 2]);

        Dbq = squeeze(batchop('mult', B, permute(Lz * Rbq0, [1 3 2]), 'T', 'N'));
        
        gradBdry = dot(Dbq.', egrad(meshData.bdryIdx, :), 2);
        gradInt = squeeze(batchop('mult', permute(egrad(meshData.intIdx, :), [3 2 1]), [Dxq Dyq Dzq])).';
        grad = [gradBdry; gradInt(:)];
    end
end

function stop = saveInfo(~, optimValues, ~)
    it = optimValues.iteration + 1;
    info(it).cost = optimValues.fval;
    info(it).gradnorm = norm(optimValues.gradient(:));
    info(it).time = toc;
    stop = info(it).gradnorm < 1e-6;
end

function [eulBdry, eulInt] = splitBdryInt(eul)
    eulBdry = eul(1:nb);
    eulInt = reshape(eul(nb+1:end), ni, 3);
end

function [RzRyRxq0, BRbq0, Rbq0, RyRxq0, Rxq0] = q0RotatedBy(eulBdry, eulInt)
    Rbq0 = rotateZ(eulBdry, q0(:, meshData.bdryIdx));
    BRbq0 = squeeze(batchop('mult', B, permute(Rbq0, [1 3 2]), 'T', 'N'));
    
    Rxq0 = rotateX(eulInt(:, 1), q0(:, meshData.intIdx));
    RyRxq0 = rotateY(eulInt(:, 2), Rxq0);
    RzRyRxq0 = rotateZ(eulInt(:, 3), RyRxq0);
end

function q = rotateZ(eulZ, q)
    eulZ = reshape(eulZ, 1, []) .* (-4:4).';
    q = cos(eulZ) .* q - sin(eulZ) .* flipud(q);
end

function q = rotateY(eulY, q)
    q = YZ' * rotateZ(eulY, YZ * q);
end

function q = rotateX(eulX, q)
    q = XZ' * rotateZ(eulX, XZ * q);
end

end