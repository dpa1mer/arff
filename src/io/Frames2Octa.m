function q = Frames2Octa(frames)
% Convert 3x3xn orthogonal matrices to octahedral coefficients.

n = size(frames, 3);
axang = rotm2axang(frames);
axang = axang(:, 4) .* axang(:, 1:3);

[~, ~, ~, YZ] = LoadSO3Generators_Y4;
q = ExpSO3(axang, repmat(OctaZAligned(0), [1, n]), YZ);

end
