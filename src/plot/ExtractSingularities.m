function [singTet, singTri, singTriType, singPoints, singEdges] ...
    = ExtractSingularities(frames, tetra)

persistent octa octaFlat eyeIdx
if isempty(octa)
    octa = cat(3, eye(3), axang2rotm([eye(3) repmat(pi/2, 3, 1)]));
    octa = reshape(multiprod(octa, permute(octa, [1 2 4 3])), 3, 3, []);
    octa = reshape(multiprod(octa, permute(octa, [1 2 4 3])), 9, []);
    octa = uniquetol(octa', 'ByRows', true);
    eyeIdx = find(ismembertol(octa, reshape(eye(3), 1, 9), 'ByRows', true));
    octa = permute(reshape(octa', 3, 3, []), [1 2 4 3]);
    octaFlat = reshape(octa, 9, []);
end

%% Find holonomy to determine singular triangles

[triangles, ~, tetTriIdx] = ...
    unique(sort([tetra(:, 1:3); tetra(:, [1, 2, 4]); tetra(:, [1, 3, 4]); tetra(:, [2, 3, 4])], 2), 'rows');
tetTriIdx = reshape(tetTriIdx, [], 4);
nTri = size(triangles, 1);

if ismatrix(frames)
    % Frames are actually field coefficients. Can sample at a finer level.
    q = frames;
    d = size(q, 1);
    if d == 9
        fiber = OctaMBO;
    else
        fiber = OdecoMBO;
    end
    
    [edges, ~, triEdgeIdx] = unique([triangles(:, [1, 2]); triangles(:, [2, 3]); triangles(:, [3, 1])], 'rows');
    triEdgeIdx = reshape(triEdgeIdx, [], 3);
    nEdges = size(edges, 1);
    
    qEdge = reshape(q(:, edges.'), d, 2, nEdges);
    edgeSamples = [2/3 1/3; 1/3 2/3];
    qMidpoint = reshape(fiber.proj(reshape(multiprod(qEdge, edgeSamples), d, [])), d, 2, nEdges);
    qLoop = reshape(q(:, triangles.'), d, 3, nTri);
    qLoop = [qLoop(:, 1, :) qMidpoint(:, :, triEdgeIdx(:, 1)) ...
             qLoop(:, 2, :) qMidpoint(:, :, triEdgeIdx(:, 2)) ...
             qLoop(:, 3, :) qMidpoint(:, :, triEdgeIdx(:, 3))];
    fLoop = reshape(Coeff2Frames(reshape(qLoop, d, []), true), 3, 3, 9, []);
    holonomy = computeHolonomy(fLoop);
else
    holonomy = computeHolonomy(reshape(frames(:, :, triangles.'), 3, 3, 3, nTri));
end
singTri = holonomy ~= eyeIdx;
singTriType = holonomy(singTri);
singTet = any(singTri(tetTriIdx), 2);
singTet = find(singTet);
singTri = find(singTri);

if nargout > 3
    %% Find singular curve intersection with each singular triangle
    
    subtri = cat(3, [1   0   0; 0.5 0.5 0;   0.5 0   0.5], ...
                    [0   1   0; 0   0.5 0.5; 0.5 0.5 0  ], ...
                    [0   0   1; 0.5 0   0.5; 0   0.5 0.5], ...
                    [0.5 0.5 0; 0   0.5 0.5; 0.5 0   0.5]);
	loop = [1 0 0; 3/4 1/4 0; 1/2 1/2 0; 1/4 3/4 0; 
            0 1 0; 0 3/4 1/4; 0 1/2 1/2; 0 1/4 3/4; 
            0 0 1; 1/4 0 3/4; 1/2 0 1/2; 3/4 0 1/4];

    assert(~isempty(q));
    
    verts = tetra.Points;

    nSingTri = length(singTri);
    triangles = triangles(singTri, :);
    vSingTri = permute(reshape(verts(triangles, :), nSingTri, 3, 3), [2 3 4 1]);
    qSingTri = reshape(q(:, triangles.'), d, 3, 1, nSingTri);

    for k = 1:4
        vSubtri = multiprod(subtri, vSingTri);
        qSubtri = reshape(multiprod(qSingTri, multitransp(subtri)), d, 3, 4 * nSingTri);
        qSubLoop = multiprod(qSubtri, multitransp(loop));
        qSubLoopProj = fiber.proj(reshape(qSubLoop, d, []));
        fSubLoop = reshape(Coeff2Frames(qSubLoopProj, true), 3, 3, 12, 4 * nSingTri);
        
        holSubtri = reshape(computeHolonomy(fSubLoop), 4, nSingTri);
        subIsSingular = holSubtri ~= eyeIdx;
        isSingular = any(subIsSingular, 1);
        singTri = singTri(isSingular);
        triangles = triangles(isSingular, :);
        nSingTri = sum(isSingular);
        
        % Take first singular subtriangle in each triangle
        subtriMask = false(size(subIsSingular));
        [~, subtriIdx] = max(subIsSingular, [], 1, 'linear');
        subtriMask(subtriIdx) = subIsSingular(subtriIdx);
        qSingTri = reshape(qSubtri(:, :, subtriMask), d, 3, 1, nSingTri);
        vSingTri = reshape(vSubtri(:, :, subtriMask), 3, 3, 1, nSingTri);
    end

    singPoints = squeeze(mean(vSingTri, 1)).';

    singTriOrder = nan(nTri, 1);
    singTriOrder(singTri) = 1:length(singTri);
    singEdges = singTriOrder(tetTriIdx(singTet, :));
    singEdges = sort(singEdges, 2);
end


function octaIdx = computeHolonomy(vertFrames)
    vertFrames = permute(vertFrames, [1 2 4 3]);
    m = size(vertFrames, 4);
    for i = 1:m
        j = mod(i, m) + 1;
        Rij = batchop('mult', vertFrames(:, :, :, i), vertFrames(:, :, :, j), 'T', 'N');
        [~, octaIdx] = max(octaFlat' * reshape(Rij, 9, []), [], 1);
        if i == m, break; end
        vertFrames(:, :, :, j) = batchop('mult', vertFrames(:, :, :, j), octa(:, :, octaIdx), 'N', 'T');
    end
end

end