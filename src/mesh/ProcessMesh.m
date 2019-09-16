function meshData = ProcessMesh(verts, tets, bdryAngleCutoff)

if nargin < 3
    bdryAngleCutoff = 0;
end

meshData.verts = verts;
meshData.tets = tets;

meshData.nv = size(meshData.verts, 1);
meshData.tetra = triangulation(meshData.tets, meshData.verts);

[meshData.L, meshData.M] = GeometricPrimalLM(meshData.verts, meshData.tets);

bdryFaces = freeBoundary(meshData.tetra);
bdryIdx = unique(bdryFaces(:));
warning('off', 'MATLAB:triangulation:PtsNotInTriWarnId');
meshData.bdry = triangulation(bdryFaces, meshData.verts);
warning('on', 'MATLAB:triangulation:PtsNotInTriWarnId');
bdryNormals = vertexNormal(meshData.bdry, bdryIdx);
nb = length(bdryIdx);

% Remove sharp creases (dihedral angle < bdryAngleCutoff)
bdryVtxStars = vertexAttachments(meshData.bdry, bdryIdx);
bdryVtxStarSizes = cellfun(@length, bdryVtxStars);
bdryVtxStarIdx = repelem((1:nb)', bdryVtxStarSizes);
bdryVtxTriIdx = cell2mat(bdryVtxStars')';
bdryVtxStarNormals = repelem(bdryNormals, bdryVtxStarSizes, 1);
bdryVtxTriNormals = faceNormal(meshData.bdry, bdryVtxTriIdx);
bdryVtxCosAngles = dot(bdryVtxStarNormals, bdryVtxTriNormals, 2);
bdryVtxMinCosAngle = accumarray(bdryVtxStarIdx, bdryVtxCosAngles, [], @min);
bdryVtxSmooth = bdryVtxMinCosAngle >= cos(0.5 * (pi - bdryAngleCutoff));
meshData.bdryIdx = bdryIdx(bdryVtxSmooth);
meshData.bdryNormals = bdryNormals(bdryVtxSmooth, :);

meshData.intIdx = setdiff((1:meshData.nv)', meshData.bdryIdx);

if gpuDeviceCount > 0
    % Inverse iteration to find second-smallest eigenvalue L*v = lambda*M*v
    v = randn(meshData.nv, 1, 'gpuArray');
    Lg = gpuArray(meshData.L);
    Mg = gpuArray(meshData.M);
    lambda = 0;
    for i = 1:100
        [v, ~] = pcg(Lg, v, 1e-6, 1000, [], [], v);
        v = v - mean(v);
        v = v / sqrt(v' * Mg * v);
        rayleigh = (v' * Lg * v);
        if abs(rayleigh - lambda) < 1e-5
            lambda = rayleigh;
            [v, ~] = pcg(@(w) Lg * w - lambda .* w, v, 1e-6, 1000, [], [], v);
            v = v / sqrt(v' * Mg * v);
            lambda = (v' * Lg * v);
            break;
        end
        lambda = rayleigh;
    end
    meshData.lambda1L = lambda;
    clear Lg Mg;
else
    lambda = eigs(meshData.L, meshData.M, 2, 'smallestabs', 'IsSymmetricDefinite', true);
    meshData.lambda1L = lambda(2);
end

end