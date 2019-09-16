function D = AlignMat(axes, YZ)

% D(:,:,i) is the representation of a rotation taking the vector
% axes(i, :) to [0;0;1]

n = size(axes, 1);
d = size(YZ, 1);

D = ExpSO3(repelem(axes, d, 1), repmat(eye(d), 1, n), YZ, true);
D = reshape(D, d, d, n);

end

