function [L, M] = GeometricPrimalLM(verts, tets)

nv = size(verts, 1);
tets = sort(tets, 2);
nt = size(tets, 1);

%% Compute Laplacian

tetToEdge = nchoosek(1:4, 2);
tetEdges = reshape(tets(:, tetToEdge), nt, 6, 2);
tetEdgesOpp = flip(tetEdges, 2);

v0 = verts(tetEdges(:, :, 1), :);
v1 = verts(tetEdges(:, :, 2), :);
w0 = verts(tetEdgesOpp(:, :, 1), :);
w1 = verts(tetEdgesOpp(:, :, 2), :);

oppEdgeLengths = vecnorm(w1 - w0, 2, 2);
oppEdgeVecs = (w1 - w0) ./ oppEdgeLengths;
n0 = cross(oppEdgeVecs, v0 - w1, 2);
n0 = n0 ./ vecnorm(n0, 2, 2);
n1 = cross(oppEdgeVecs, v1 - w1, 2);
n1 = n1 ./ vecnorm(n1, 2, 2);
t1 = cross(oppEdgeVecs, n1, 2);

% Make sure we are consistently using the positively oriented angle
alpha = abs(atan2(dot(n0, t1, 2), dot(n0, n1, 2)));
oppositeCotans = cot(alpha);

Lij = oppEdgeLengths .* oppositeCotans / 6;
L = sparse(tetEdges(:, :, 1), tetEdges(:, :, 2), Lij, nv, nv);
L = L + L';
L = diag(sum(L, 1)) - L;

%% Compute Mass Matrix

Mij = abs(dot(v1 - v0, cross(w0 - v0, w1 - v0, 2), 2)) / 120;
Mii = repmat(TetVolumes(verts, tets) / 10, [1 4]);

M = sparse(tetEdges(:, :, 1), tetEdges(:, :, 2), Mij, nv, nv);
M = M + M' + sparse(tets, tets, Mii, nv, nv);

end

