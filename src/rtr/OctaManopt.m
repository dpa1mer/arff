function [q, q0, info] = OctaManopt(meshData, q0, saveIterates, gpuflag, useCombinatorialLaplacian)

if nargin < 3
    saveIterates = false;
end

if nargin < 4
    gpuflag = gpuDeviceCount > 0;
end

if nargin < 5
    useCombinatorialLaplacian = false;
end

%% Initial value

if nargin < 2 % Initialize randomly
    q0 = [];
elseif gpuflag
    q0 = gpuArray(q0);
end

%% Manifold Optimization

octa = OctahedralBundleFactory(meshData.nv, meshData.bdryIdx, meshData.bdryNormals, gpuflag);
problem.M = octa;

A = meshData.L;
if useCombinatorialLaplacian
    E = edges(meshData.tetra);
    edgeGraph = graph(E(:, 1), E(:, 2));
    A = laplacian(edgeGraph);
end
invDiagA = full(diag(A)).^(-1);
if gpuflag
    A = gpuArray(A);
    invDiagA = gpuArray(invDiagA);
end

problem.cost = @myDirichlet;
problem.grad = @myGradient;
problem.hess = @myHessian;
problem.precon = @myPrecon;

function store = prepare(q, store)
    if ~isfield(store, 'Aq')
        store.Aq = (A * q')';
    end
    if ~isfield(store, 'tangentbasis')
        store.tangentbasis = octa.tangentbasis(q);
    end
    if ~isfield(store, 'rgrad')
        store.rgrad = octa.egrad2rgrad(q, store.Aq, store.tangentbasis);
    end
    if ~isfield(store, 'LxyzTegrad')
        store.LxyzTegrad = octa.mulLxyzT(store.Aq);
    end
end

function [cost, store] = myDirichlet(q, store)
    store = prepare(q, store);
    cost = 0.5 * q(:)' * store.Aq(:);
end

function [grad, store] = myGradient(q, store)
    store = prepare(q, store);
    grad = store.rgrad;
end

function [hess, store] = myHessian(q, s, store)
    store = prepare(q, store);
    sAmbient = octa.tangent2ambient(q, s, store.tangentbasis);
    ehess = (A * sAmbient')';
    hess = octa.ehess2rhess(q, store.Aq, ehess, s, sAmbient, store.rgrad, store.LxyzTegrad, store.tangentbasis);
end

function Ps = myPrecon(~, s)
    Ps = invDiagA .* s;
end

% figure; whitebg('white');
% checkgradient(problem);
% pause;
% 
% figure; whitebg('white');
% checkhessian(problem);
% pause;

opts = struct;
if saveIterates
    opts.statsfun = statsfunhelper('q', @(q) gather(q));
end

[q, ~, info] = trustregions(problem, sqrt(3/20) * q0, opts);

% Rescale to normalize q
q = gather(sqrt(20/3) * q);

end
