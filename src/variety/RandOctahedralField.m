function q = RandOctahedralField(n, alignIdx, alignAxes, gpuflag)

if nargin < 4
    gpuflag = false;
end

dataType = 'double';
if gpuflag
    dataType = 'gpuArray';
end

q = OctaZAligned(2*pi*rand(n, 1, dataType));
axes = randn(n, 3, dataType);
axes = axes ./ sqrt(dot(axes, axes, 2));
axes(alignIdx, :) = alignAxes;
D = OctaAlignMat(axes);
q = multiprod(multitransp(D), q, [1 2], 1);

end

