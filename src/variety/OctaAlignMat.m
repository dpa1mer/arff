function [RotMatStacked, AlignMatStacked] = OctaAlignMat(normals)

% RotMatStacked(:,:,i) is the representation of a rotation taking the vector
% normal(i, :) to [0;0;1]

[~, ~, ~, YZ] = LoadSO3Generators_Y4;

RotMatStacked = AlignMat(normals, YZ);
if nargout > 1
    AlignMatStacked = RotMatStacked(2:8, :, :);
end

end

