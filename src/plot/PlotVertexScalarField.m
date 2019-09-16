function PlotVertexScalarField(tetra, values)
% Plots a piecewise-linear scalar field defined by its values on the
% vertices of a tetrahedral mesh.

[lBounds, uBounds] = bounds(tetra.Points, 1);

% Define grid resolution based on min edge length
E = edges(tetra);
edgeLengths = vecnorm(tetra.Points(E(:, 1), :) - tetra.Points(E(:, 2), :), 2, 2);
voxelSize = min(edgeLengths) / 2;

[xGrid, yGrid, zGrid] = meshgrid(lBounds(1):voxelSize:uBounds(1), ...
                                 lBounds(2):voxelSize:uBounds(2), ...
                                 lBounds(3):voxelSize:uBounds(3));
[tetIdx, baryCoords] = pointLocation(tetra, [xGrid(:), yGrid(:), zGrid(:)]);
insideIdx = find(~isnan(tetIdx));
tetIdx = tetIdx(insideIdx);
baryCoords = baryCoords(insideIdx, :);
vertVals = values(tetra(tetIdx, :));
gridVals = zeros(size(xGrid));
gridVals(insideIdx) = dot(baryCoords, vertVals, 2);

volshow(gridVals, 'Renderer', 'MaximumIntensityProjection', ...
        'BackgroundColor', 'white', ...
        'Colormap', inferno(256), ...
        'Alphamap', [zeros(32, 1); linspace(0, 1, 192).'; ones(32, 1)]);

% figure; sl = slice(xq, yq, zq, gridVals, 0, 0, 0);
% set(sl, 'EdgeColor', 'none');

% figure;
% p = patch(isosurface(xGrid, yGrid, zGrid, gridVals, 0.15));
% isonormals(xGrid, yGrid, zGrid, gridVals, p);
% p.EdgeColor = 'none';
% p.FaceColor = 'red';
% camlight;
% lighting gouraud;
% axis tight;
% axis vis3d;
% view(3);

end

