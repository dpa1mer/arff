function [q, frames] = RayInit(meshData)

normals = zeros(meshData.nv, 3);
normals(meshData.bdryIdx, :) = meshData.bdryNormals;
tets = int32(sortrows(meshData.tets) - 1).';
frames = RayInitHelper(normals.', tets);
frames = reshape(frames, [3 3 meshData.nv]);
q = Frames2Octa(frames);

end