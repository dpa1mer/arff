function ExportIntegralCurves(filename, frames, tetra, varargin)

p = inputParser;
p.KeepUnmatched = true;
addParameter(p, 'ClipPlane', []);
parse(p, varargin{:});

curveHeads = IntegralCurves(frames, tetra, varargin{:});
if ~isempty(p.Results.ClipPlane)
    clip = reshape(p.Results.ClipPlane, [1 1 3]);
    curveHeads(:, any(sum(curveHeads .* clip, 3) > 1, 1), :) = nan;
end

curveVerts = reshape(curveHeads(isfinite(curveHeads)), [], 3);
curveVerts = [array2table(repmat('v', length(curveVerts), 1)), array2table(curveVerts)];
writetable(curveVerts, filename, 'FileType', 'text', 'Delimiter', ' ', 'WriteVariableNames', false);

curveIdx = nan(size(curveHeads, 1), size(curveHeads, 2));
curveIdx(all(isfinite(curveHeads), 3)) = 1 : sum(all(isfinite(curveHeads), 3), 'all');
curveIdxCell = num2cell(curveIdx.', 2);
curveIdxCell = cellfun(@(i) i(isfinite(i)), curveIdxCell, 'UniformOutput', false);
curveEdgesCell = cellfun(@(i) [i(1:end-1).' i(2:end).'], curveIdxCell, 'UniformOutput', false);
curveEdges = cell2mat(curveEdgesCell);
curveEdges = [array2table(repmat('f', length(curveEdges), 1)), array2table(curveEdges)];

tempFilename = [filename '.temp'];
writetable(curveEdges, tempFilename, 'FileType', 'text', 'Delimiter', ' ', 'WriteVariableNames', false);
system(['cat ' tempFilename ' >> ' filename]);
system(['rm ' tempFilename]);

end