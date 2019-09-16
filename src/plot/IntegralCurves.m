function [curveHeads, curveColor] = IntegralCurves(frames, tetra, varargin)

nt = size(tetra, 1);

[lb, ub] = bounds(tetra.Points, 1);
defaultCurveLength = 0.25 * max(ub - lb);

p = inputParser;
p.KeepUnmatched = true;
scalarFieldValidator = @(x) assert(length(x) == size(frames, 3));
addParameter(p, 'NumSeeds', 1000);
addParameter(p, 'Prune', true);
addParameter(p, 'ColorField', [], scalarFieldValidator);
addParameter(p, 'CurveLength', defaultCurveLength);
parse(p, varargin{:});

nSeeds = p.Results.NumSeeds;
prune = p.Results.Prune;
colorField = p.Results.ColorField;
interpColor = ~isempty(colorField);
curveLength = p.Results.CurveLength;

%% Choose dt and nSteps based on size of elements
E = edges(tetra);
edgeLengths = vecnorm(tetra.Points(E(:,1), :) - tetra.Points(E(:, 2), :), 2, 2);
dt = 0.5 * (mean(edgeLengths) - std(edgeLengths));

nSteps = int32(curveLength / dt);

%% Seed Points Uniformly in Volume

curveHeads = nan(nSteps, nSeeds, 3);
curveColor = nan(nSeeds, nSteps);

vol = TetVolumes(tetra.Points, tetra.ConnectivityList);
randTets = randsample(nt, nSeeds, true, vol);
randBaryCoords = rand(nSeeds, 4);
randBaryCoords = randBaryCoords ./ sum(randBaryCoords, 2);
curveHeads(1, :, :) = barycentricToCartesian(tetra, randTets, randBaryCoords);
if interpColor
    curveColor(:, 1) = dot(randBaryCoords, colorField(tetra(randTets, :)), 2);
end

curveVel = randn(nSeeds, 3);
curveVel = curveVel ./ vecnorm(curveVel, 2, 2);

%% Trace Curves

frames = [frames -frames];
frames = multitransp(frames);

nCurves = nSeeds;
insideIdx = 1:nCurves;

for k = 2:nSteps
    oldInsideIdx = insideIdx;
    [tetIdx, baryCoords] = pointLocation(tetra, permute(curveHeads(k - 1, insideIdx, :), [2 3 1]));
    insideIdx = find(~isnan(tetIdx));
    if isempty(insideIdx)
        break;
    end
    nCurves = size(insideIdx, 1);
    tetIdx = tetIdx(insideIdx);
    baryCoords = baryCoords(insideIdx, :);
    insideIdx = oldInsideIdx(insideIdx);
    
    neighVertIdx = tetra(tetIdx, :);
    neighVertFrames = reshape(frames(:, :, neighVertIdx), 6, 3, nCurves, 4);
    
    if interpColor
        curveColor(insideIdx, k) = dot(baryCoords, reshape(colorField(neighVertIdx), size(neighVertIdx)), 2);
    end
    
    [~, neighVertMatch] = max(multiprod(neighVertFrames, curveVel(insideIdx, :)', [1 2], 1));
    neighVertMatch = repmat(permute(neighVertMatch, [2 3 1]), [1 1 3]);
    neighVertIdx = repmat(neighVertIdx, [1 1 3]);
    xyz = repmat(permute((1:3), [1 3 2]), [nCurves, 4, 1]);
    
    idx = sub2ind(size(frames), neighVertMatch, xyz, neighVertIdx);
    neighVertVel = frames(idx);
    
    curveVel(insideIdx, :) = squeeze(multiprod(baryCoords, neighVertVel, 2, [2 3]));
    curveVel(insideIdx, :) = curveVel(insideIdx, :) ./ vecnorm(curveVel(insideIdx, :), 2, 2);
    
    curveHeads(k, insideIdx, :) = curveHeads(k - 1, insideIdx, :) + dt * permute(curveVel(insideIdx, :), [3 1 2]);
end

%% Prune curves

if prune
    retainIdx = find(sum(all(isfinite(curveHeads), 3), 1) > min(5, 0.1 * nSteps));
    curveHeads = curveHeads(:, retainIdx, :);
    if interpColor
        curveColor = curveColor(retainIdx, :);
    end
end

end

