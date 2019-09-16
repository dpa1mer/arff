function VisualizeFrameField(frames, energyDensity, tetra, bdry)

fig = figure;

frames = frames ./ vecnorm(frames, 2, 1);

% Singularities only
% singularIdx = find(energyDensity > mean(energyDensity) + std(energyDensity));
% singularTets = vertexAttachments(tetra, singularIdx)';
% singularTets = unique(cell2mat(singularTets)');
singularTets = ExtractSingularities(frames, tetra);
warning('off', 'MATLAB:triangulation:PtsNotInTriWarnId');
singularTetra = triangulation(tetra(singularTets, :), tetra.Points);
warning('on', 'MATLAB:triangulation:PtsNotInTriWarnId');
ax1 = subplot(1, 2, 1);
PlotIntegralCurves(frames, singularTetra, 'NumSeeds', 10000, 'Prune', true, 'ColorField', energyDensity);
hold on;
trisurf(bdry, 'FaceColor', 'black', 'EdgeColor', 'none', 'FaceAlpha', 0.01);
view(3);
axis image vis3d off;

% Integral Curves
ax2 = subplot(1, 2, 2); PlotIntegralCurves(frames, tetra, 'NumSeeds', 10000, 'Prune', true);
view(3);
axis image vis3d off;

Link = linkprop([ax1 ax2], {'CameraUpVector', 'CameraPosition', 'CameraTarget'});
setappdata(fig, 'TheLink', Link);

set(fig, 'color', 'white');

end

