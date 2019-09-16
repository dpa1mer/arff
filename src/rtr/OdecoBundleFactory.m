function M = OdecoBundleFactory(n, bdryIdx, bdryNormals, gpuflag)

if nargin < 4
    gpuflag = false;
end

%% Set up parameters

odecoMat = LoadOdecoMatsSph;
odecoMat = cell2mat(permute(odecoMat, [2 3 1]));
odecoMatStacked   = reshape(odecoMat, 15, [])';
odecoMatFlattened = reshape(odecoMat, 15^2, 27);

[Lx4, Ly4, Lz4, YZ4] = LoadSO3Generators_Y4;
[Lx2, Ly2, Lz2, YZ2] = LoadSO3Generators_Y2;
Lx = blkdiag(0, Lx2, Lx4);
Ly = blkdiag(0, Ly2, Ly4);
Lz = blkdiag(0, Lz2, Lz4);
Lxyz = cat(3, Lx, Ly, Lz);

dataType = 'double';
if gpuflag
    dataType = 'gpuArray';
    bdryIdx = gpuArray(bdryIdx);
    bdryNormals = gpuArray(bdryNormals);
    Lxyz = gpuArray(Lxyz);
    YZ2 = gpuArray(YZ2);
    YZ4 = gpuArray(YZ4);
end
Id = eye(15, dataType);

nb = length(bdryIdx);
intIdx = setdiff(1:n, bdryIdx).';
ni = length(intIdx);

if nb > 0
    [~, BdryAlignStacked] = OdecoAlignMat(bdryNormals);
    BdryProj = Id - batchop('mult', BdryAlignStacked, BdryAlignStacked, 'T', 'N');
end

%% Manifold Interface

M.dim = @() 6 * ni + 3 * nb;

M.inner = @(q, t1, t2) t1(:)' * t2(:);

M.norm = @(q, t) norm(t(:));

% M.typicaldist = pi / 2 * sqrt(3 * n);

M.mulO = @(q) reshape(odecoMatStacked * q, 15, 27, n);

M.proj = @projection;
function [v, NqM, yO] = projection(q, v0, NqM)
    if nargin < 3
        NqM.Oq = M.mulO(q);
        NqM.basis = NqM.Oq;
        NqM.basis(:, :, bdryIdx) = batchop('mult', BdryProj, NqM.basis(:, :, bdryIdx));
        NqM.pinv(:, :, bdryIdx) = batchop('pinv', NqM.basis(:, :, bdryIdx), 4);
        NqM.pinv(:, :, intIdx) = batchop('pinv', NqM.basis(:, :, intIdx), 9);
        NqM.projTang = Id - batchop('mult', NqM.Oq, NqM.pinv);
        NqM.projTang(:, :, bdryIdx) = batchop('mult', BdryProj, NqM.projTang(:, :, bdryIdx));
    end
    
    v = squeeze(batchop('mult', NqM.projTang, permute(v0, [1 3 2])));
    
    if nargout > 2
        dualY = batchop('mult', NqM.pinv, permute(v0, [1 3 2]));
        yO = reshape(odecoMatFlattened * squeeze(dualY), 15, 15, n);
    end
end

M.tangent = @(q, v) v; %projection;

M.egrad2rgrad = M.proj;

M.ehess2rhess = @ehess2rhess;
function rhess = ehess2rhess(q, ~, ehess, v, NqM, gradYO)
    rhess = ehess - squeeze(batchop('mult', gradYO, permute(v, [1 3 2])));
    rhess = M.proj(q, rhess, NqM);
end

M.retr = @retraction;
function q = retraction(q0, v, t)
    if nargin == 3
        tv = t * v;
    else
        tv = v;
    end
    
    rotBasis = permute(multiprod(Lxyz, q0), [1 3 2]);
    rotPart = batchop('leastsq', rotBasis, permute(tv, [1 3 2]));
    
    linearPart = tv - squeeze(batchop('mult', rotBasis, rotPart));
    rotPart = squeeze(rotPart);
    
    q = q0 + linearPart;
    q(2:6,:) = ExpSO3(rotPart', q(2:6, :), YZ2);
    q(7:end,:) = ExpSO3(rotPart', q(7:end, :), YZ4);
end

M.rand = @rand;
function q = rand()
    q = Octa2Odeco(RandOctahedralField(n, bdryIdx, bdryNormals, gpuflag));
end

M.randvec = @randvec;
function s = randvec(q)
    s = M.proj(q, randn(15, n, dataType));
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

M.zerovec = @(q) zeros(15, n, dataType);

% M.transp = @(q1, q2, s) s; % Project?

end

