function PlotFrames(frames, centers, varargin)

n = size(frames, 3);
assert(n == size(centers, 1));

p = inputParser;
scalarFieldValidator = @(x) assert(length(x) == n);
addParameter(p, 'GlobalScale', 1);
addParameter(p, 'ScaleField', [], scalarFieldValidator);
addParameter(p, 'ColorField', [], scalarFieldValidator);
addParameter(p, 'NormalColorRot', []);
addParameter(p, 'ColorMap', hot);
addParameter(p, 'LineWidth', 1);
addParameter(p, 'PlotCubes', false);
parse(p, varargin{:});

if ~isempty(p.Results.ScaleField)
    frames = frames .* permute(p.Results.ScaleField, [3 2 1]);
end

if p.Results.PlotCubes
    cube = cat(3, [0 0 0; 1 0 0; 1 1 0; 0 1 0], ...
                  [0 0 0; 1 0 0; 1 0 1; 0 0 1], ...
                  [0 0 0; 0 1 0; 0 1 1; 0 0 1], ...
                  [1 0 0; 1 1 0; 1 1 1; 1 0 1], ...
                  [0 1 0; 0 1 1; 1 1 1; 1 1 0], ...
                  [0 0 1; 1 0 1; 1 1 1; 0 1 1]);
	cubeNormals = [0 0 -1; 0 -1 0; -1 0 0; 1 0 0; 0 1 0; 0 0 1];
              
    % Scale by shortest distance between points
    [~, d] = knnsearch(centers, centers, 'K', 2);
    cube = (cube - 0.5) * (min(d(:, 2)) / sqrt(3)) * p.Results.GlobalScale;
    
    cube = permute(cube, [2 1 4 3]);
    cubes = multiprod(frames, cube) + permute(centers, [2 3 1]);
    cubes = permute(cubes, [2 4 3 1]);
    cubes = reshape(cubes, 4, [], 3);
    
    if ~isempty(p.Results.ColorField)
        colors = repelem(p.Results.ColorField, 6);
    elseif ~isempty(p.Results.NormalColorRot)
        normals = multiprod(frames, permute(cubeNormals, [2 3 4 1]));
        normals = reshape(permute(normals, [4 3 1 2]), [], 3);
        normals = normals ./ vecnorm(normals, 2, 2);
        rot = p.Results.NormalColorRot;
        colors = abs(normals * rot);
    else
        colors = repmat(0.9, 6 * n, 3);
    end
    patch(cubes(:, :, 1), cubes(:, :, 2), cubes(:, :, 3), permute(colors, [1 3 2]), ...
          'EdgeColor', 'none', 'FaceLighting', 'flat', 'FaceColor', 'flat');
else
    centers = reshape(repmat(centers', [6, 1]), 3, [])';
    v = reshape(frames, 9, n);
    v = [v; -v];
    v = reshape(v, 3, [])';

    q = quiver3(centers(:, 1), centers(:, 2), centers(:, 3), v(:, 1), v(:, 2), v(:, 3), p.Results.GlobalScale, '-w');
    q.ShowArrowHead = 'off';
    q.AlignVertexCenters = 'on';
    q.LineWidth = p.Results.LineWidth;
    
    if ~isempty(p.Results.ColorField)
        % Color a la
        % https://stackoverflow.com/questions/29632430/quiver3-arrow-color-corresponding-to-magnitude
        colorMap = p.Results.ColorMap;
        colorField = repmat(p.Results.ColorField', 6, 1);
        [~, ~, colorIdx] = histcounts(colorField, size(colorMap, 1));
        cmap = uint8(ind2rgb(colorIdx(:), colorMap) * 255);
        cmap(:, :, 4) = 255; % Opaque
        cmap = permute(repmat(cmap, [1 2 1]), [2 1 3]);
        q.Tail.ColorData = reshape(cmap, [], 4)';
        q.Tail.ColorBinding = 'interpolated';
    end
end

end