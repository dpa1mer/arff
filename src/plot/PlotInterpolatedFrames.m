function PlotInterpolatedFrames(q, tetra, samples, varargin)

d = size(q, 1);

p = inputParser;
addParameter(p, 'ColorField', []);
addParameter(p, 'NormalColorRot', []);
parse(p, varargin{:});

[tetIdx, baryCoords] = pointLocation(tetra, samples);
insideIdx = find(~isnan(tetIdx));
if isempty(insideIdx)
    return;
end
samples = samples(insideIdx, :);
tetIdx = tetIdx(insideIdx);
baryCoords = baryCoords(insideIdx, :);

qVerts = reshape(q(:, tetra(tetIdx, :)'), d, 4, []);
qSamples = squeeze(multiprod(permute(baryCoords, [3 2 1]), qVerts, 2, 2));

if d == 9
    projector = OctaMBO;
else
    projector = OdecoMBO;
end
qSamples = projector.proj(qSamples);
fSamples = Coeff2Frames(qSamples, false);

if ~isempty(p.Results.ColorField)
    colors = p.Results.ColorField(qSamples);
    PlotFrames(fSamples, samples, 'GlobalScale', 1, 'PlotCubes', true, 'ColorField', colors);
elseif ~isempty(p.Results.NormalColorRot)
    PlotFrames(fSamples, samples, 'GlobalScale', 1, 'PlotCubes', true, 'NormalColorRot', p.Results.NormalColorRot);
else
    PlotFrames(fSamples, samples, 'GlobalScale', 1, 'PlotCubes', true);
end


end

