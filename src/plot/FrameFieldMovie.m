function FrameFieldMovie(meshData, solveInfo, outFile)

vid = VideoWriter(outFile);
open(vid);

fig = figure('units','normalized','outerposition',[0 0 1 1]); 

for i = 1:numel(solveInfo)
    cla;
    
    q = solveInfo(i).q;
    if size(q, 1) == 9 % Octahedral
        energyDensity = dot(q.', meshData.L * q.', 2);
    else % Odeco
        energyDensity = dot(q(7:15, :).', meshData.L * q(7:15, :).', 2);
    end
    frames = Coeff2Frames(q, true);
    
    singularTets = ExtractSingularities(frames, meshData.tetra);
    warning('off', 'MATLAB:triangulation:PtsNotInTriWarnId');
    singularTetra = triangulation(meshData.tets(singularTets, :), meshData.verts);
    warning('on', 'MATLAB:triangulation:PtsNotInTriWarnId');
    PlotIntegralCurves(frames, singularTetra, 'NumSeeds', 10000, 'Prune', true, 'ColorField', energyDensity);
    
    hold on;
    trisurf(meshData.bdry, 'FaceColor', 'black', 'EdgeColor', 'none', 'FaceAlpha', 0.01);
    if i==1
        view(3);
        axis image vis3d off manual;
        set(fig, 'color', 'black');
    end
    
    vidFrame = getframe(gca);
    writeVideo(vid, vidFrame);
end
close(vid);

end