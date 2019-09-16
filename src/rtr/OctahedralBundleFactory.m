function M = OctahedralBundleFactory(n, bdryIdx, bdryNormals, gpuflag)

%% Set up parameters

if nargin < 4
    gpuflag = false;
end

[Lx, Ly, Lz, YZ] = LoadSO3Generators_Y4;
Lxyz = cat(3, Lx, Ly, Lz);

dataType = 'double';
if gpuflag
    dataType = 'gpuArray';
    bdryIdx = gpuArray(bdryIdx);
    bdryNormals = gpuArray(bdryNormals);
    Lxyz = gpuArray(Lxyz);
    YZ = gpuArray(YZ);
end

nb = length(bdryIdx);
intIdx = setdiff(1:n, bdryIdx);

LxyzT = -Lxyz;

%% Manifold Interface

M.dim = @() 3 * n;

M.inner = @(q, t1, t2) t1(:)' * t2(:);

M.norm = @(q, t) norm(t(:));

M.typicaldist = pi / 2 * sqrt(3 * n);

M.mulLxyz = @(v) multiprod(Lxyz, v);
M.mulLxyzT = @(v) multiprod(LxyzT, v);

M.tangentbasis = M.mulLxyz;

% 9 x n -> n x 3
M.proj = @projection;
function s = projection(q, v, Lxyzq)
    if nargin < 3
        Lxyzq = M.tangentbasis(q);
    end
    s = squeeze(sum(v .* Lxyzq, 1));
    
    % Constrain boundary frames to rotate around normals
    s(bdryIdx, :) = dot(s(bdryIdx, :), bdryNormals, 2) .* bdryNormals;
end

M.tangent = @tangentialize;
function s = tangentialize(~, s)
    s(bdryIdx, :) = dot(s(bdryIdx, :), bdryNormals, 2) .* bdryNormals;
end

% n x 3 -> 9 x n
M.tangent2ambient = @tangent2ambient;
function v = tangent2ambient(q, s, Lxyzq)
    if nargin < 3
        Lxyzq = M.tangentbasis(q);
    end
    v = multiprod(s, Lxyzq, 2, 3);
end

M.egrad2rgrad = M.proj;

M.ehess2rhess = @ehess2rhess;
function rhess = ehess2rhess(q, egrad, ehess, s, sAmbient, rgrad, LxyzTegrad, Lxyzq)
    if nargin < 5
        sAmbient = M.tangent2ambient(q, s);
    end
    if nargin < 6
        rgrad = M.egrad2rgrad(q, egrad);
    end
    if nargin < 7
        LxyzTegrad = M.mulLxyzT(egrad);
    end
    if nargin < 8
        Lxyzq = M.tangentbasis(q);
    end
    
    rhess = M.proj(q, ehess, Lxyzq);
    
    % Interior
    intTerm = (1/2)*fastCross(rgrad, s) ...
            + squeeze(sum(sAmbient .* LxyzTegrad, 1));
    rhess(intIdx, :) = rhess(intIdx, :) + intTerm(intIdx, :);
    
    % Boundary
    bdryBasis = multiprod(bdryNormals, LxyzTegrad(:, bdryIdx, :), 2, 3);
    bdryTerm = squeeze(dot(sAmbient(:, bdryIdx), bdryBasis, 1))' .* bdryNormals;
    rhess(bdryIdx, :) = rhess(bdryIdx, :) + bdryTerm;
end

function c = fastCross(a, b)
    c(:, [3 1 2]) = a .* b(:, [2 3 1]) - a(:, [2 3 1]) .* b;
end

M.retr = @exponential;

M.exp = @exponential;
function q = exponential(q0, s, t)
    if nargin == 3
        ts = t * s;
    else
        ts = s;
    end
    
    q = ExpSO3(ts, q0, YZ);
end

M.rand = @() sqrt(3/20) * RandOctahedralField(n, bdryIdx, bdryNormals, gpuflag);

M.randvec = @randvec;
function s = randvec(~)
    s = randn(n, 3, dataType);
    s(bdryIdx, :) = randn(nb, 1, dataType) .* bdryNormals;
    s = s / norm(s(:));
end

if gpuflag
    M.lincomb = @gpulincomb;
else
    M.lincomb = @matrixlincomb;
end
function v = gpulincomb(~, a1, d1, a2, d2)
    if nargin == 3
        v = a1 * d1;
    elseif nargin == 5
        v = arrayfun(@gpulincombElt, a1, d1, a2, d2);
    else
        error('gpulincomb takes either 3 or 5 inputs.');
    end
end
function v = gpulincombElt(a1, d1, a2, d2)
    v = a1 * d1 + a2 * d2;
end

M.zerovec = @(q) zeros(n, 3, dataType);

% M.transp = @(q1, q2, s) s;

end

