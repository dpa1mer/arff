function PlotIntegralCurves(frames, tetra, varargin)

p = inputParser;
p.KeepUnmatched = true;
addParameter(p, 'ColorField', []);
parse(p, varargin{:});
interpColor = ~isempty(p.Results.ColorField);

[curveHeads, curveColor] = IntegralCurves(frames, tetra, varargin{:});
nCurves = size(curveHeads, 2);
nSteps = size(curveHeads, 1);

% Hacking with patch is faster than creating a bunch of lines
if interpColor
    colormap(gca, inferno);
    colors = reshape([curveColor'; nan(1, nCurves)], [], 1);
    [~, perm] = sort(colors(isfinite(colors)));
    equalizedColors(perm) = 1:length(perm);
    colors(isfinite(colors)) = equalizedColors;
else
    colormap(gca, lines);
    colors = repmat(randi(size(lines, 1), 1, nCurves), nSteps, 1);
    colors = reshape([colors; nan(1, nCurves)], [], 1);
end
curves = reshape([curveHeads; nan(1, nCurves, 3)], [], 3);
p = patch([curves(:, 1); nan], [curves(:, 2); nan], [curves(:, 3); nan], [colors; nan]);
p.LineWidth = 1.5;
p.EdgeColor = 'interp';
p.FaceColor = 'none';

end

