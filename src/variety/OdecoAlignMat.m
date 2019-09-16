function [RotMatStacked, AlignMatStacked] = OdecoAlignMat(normals)

n = size(normals, 1);

[~, ~, ~, YZ2] = LoadSO3Generators_Y2;
[~, ~, ~, YZ4] = LoadSO3Generators_Y4;
D2 = AlignMat(normals, YZ2);
D4 = AlignMat(normals, YZ4);

% RotMatStacked(:,:,i) is the representation of a rotation taking the vector
% normal(i, :) to [0;0;1]
RotMatStacked = [ones(1, 1, n)     zeros(1, 14, n);
                 zeros(5, 1, n) D2  zeros(5, 9, n);
                 zeros(9, 6, n)                D4];

if nargout > 1
    AlignMatStacked = [(sqrt(7)/5) * ones(1, 1, n),  sqrt(5/7) * D2(3, :, :), (sqrt(7)/35) * D4(5, :, :);
                       (5*sqrt(2))^(-1) * ones(1, 1, n),  zeros(1, 5, n),   -7/(5*sqrt(2)) * D4(5, :, :);
                       zeros(2, 1, n), D2([2,4], :, :), zeros(2, 9, n);
                       zeros(4, 6, n), D4([2,4,6,8], :, :)];
end

end

