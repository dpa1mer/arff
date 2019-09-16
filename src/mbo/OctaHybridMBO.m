function octahedral = OctaHybridMBO

octahedral.dim = 9;

%% Projection

octahedral.proj = @rtrProj;
function q = rtrProj(q0)
    if ~isfield(octahedral, 'bundle')
        octahedral.bundle = OctahedralBundleFactory(size(q0, 2), [], [], isa(q0, 'gpuArray'));
    end
    
    % Normalize for proper scaling
    q0 = q0 ./ vecnorm(q0, 2, 1);
    
    problem.M = octahedral.bundle;
    problem.cost = @myDistance;
    problem.grad = @myGradient;
    problem.hess = @myHessian;

    function store = prepare(q, store)
        if ~isfield(store, 'tangentbasis')
            store.tangentbasis = octahedral.bundle.tangentbasis(q);
        end
        if ~isfield(store, 'rgrad')
            store.rgrad = octahedral.bundle.egrad2rgrad(q, -q0, store.tangentbasis);
        end
        if ~isfield(store, 'LxyzTegrad')
            store.LxyzTegrad = octahedral.bundle.mulLxyzT(-q0);
        end
    end

    function [cost, store] = myDistance(q, store)
        store = prepare(q, store);
        cost = -q0(:)' * q(:);
    end

    function [grad, store] = myGradient(q, store)
        store = prepare(q, store);
        grad = store.rgrad;
    end

    function [hess, store] = myHessian(q, s, store)
        store = prepare(q, store);
        sAmbient = octahedral.bundle.tangent2ambient(q, s, store.tangentbasis);
        hess = octahedral.bundle.ehess2rhess(q, -q0, zeros(size(q), 'like', q), s, sAmbient, store.rgrad, store.LxyzTegrad, store.tangentbasis);
    end

%     figure; whitebg('white');
%     checkgradient(problem);
%     pause;
%     
%     figure; whitebg('white');
%     checkhessian(problem);
%     pause;
    
    opts.verbosity = 0;
    q = trustregions(problem, [], opts);

    % Rescale to normalize q
    q = gather(sqrt(20/3) * q);
    
end

%% Boundary Alignment

octahedral.projAligned = @(q) sqrt(5/12) * (q ./ vecnorm(q, 2, 1));

octahedral.bdryBasis = @octaBdryBasis;
function [bdryFixed, bdryBasis] = octaBdryBasis(bdryNormals)
    BdryRotStacked = OctaAlignMat(bdryNormals);
    bdryBasis = multiprod(multitransp(BdryRotStacked), sparse([1 9], [1 2], [1, 1], 9, 2));
    bdryFixed = multiprod(multitransp(BdryRotStacked), [0 0 0 0 sqrt(7/12) 0 0 0 0]', [1 2], 1);
end

%% Initial value

octahedral.rand = @RandOctahedralField;

end
