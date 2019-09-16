function PlotSingularGraph(q, tetra)

[~, ~, ~, singPoints, singEdges] = ExtractSingularities(q, tetra);

fig = figure;
p = patch('Faces', singEdges, 'Vertices', singPoints);
p.LineWidth = 2;
view(3);
axis image vis3d off;
set(fig, 'color', 'white');

end

