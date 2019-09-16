function [q, q0, info] = OdecoManopt(meshData, q0, saveIterates, gpuflag)

if nargin < 3
    saveIterates = false;
end

if nargin < 4
    gpuflag = false;
end

%% Initial value

if nargin < 2 % Initialize randomly
    q0 = [];
elseif gpuflag
    q0 = gpuArray(q0);
end

%% Manifold Optimization

odeco = OdecoBundleFactory(meshData.nv, meshData.bdryIdx, meshData.bdryNormals, gpuflag);
problem.M = odeco;
A = meshData.L;
if gpuflag
    A = gpuArray(A);
end

% invDiagA = full(diag(A)).^(-1);
% if gpuflag
%     A = gpuArray(A);
%     invDiagA = gpuArray(invDiagA);
% end

problem.cost = @myDirichlet;
problem.grad = @myGradient;
problem.hess = @myHessian;
% problem.precon = @myPrecon;

function Ax = mulA(x)
    Ax = (A * x.').';
end

function store = prepare(q, store)
    if ~isfield(store, 'Aq')
        store.Aq = mulA(q);
    end
    if ~isfield(store, 'rgrad')
        [store.rgrad, store.NqM, store.gradYO] = odeco.egrad2rgrad(q, store.Aq);
    end
end

function [cost, store] = myDirichlet(q, store)
    store = prepare(q, store);
    cost = 0.5 * (q(:)' * store.Aq(:));
end

function [grad, store] = myGradient(q, store)
    store = prepare(q, store);
    grad = store.rgrad;
end

function [hess, store] = myHessian(q, v, store)
    store = prepare(q, store);
    ehess = mulA(v);
    hess = odeco.ehess2rhess(q, store.Aq, ehess, v, store.NqM, store.gradYO);
end

% function Ps = myPrecon(~, s)
%     Ps = invDiagA.' .* s;
% end

% figure; whitebg('white');
% checkgradient(problem);
% pause;
% 
% figure; whitebg('white');
% checkhessian(problem);
% pause;

opts.Delta_bar = sqrt(6*meshData.nv);
if saveIterates
    opts.statsfun = statsfunhelper('q', @(q) gather(q));
end

[q, ~, info] = trustregions(problem, q0, opts);

if gpuflag
    q = gather(q);
end

end
