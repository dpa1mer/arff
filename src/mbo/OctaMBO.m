function octahedral = OctaMBO

octahedral.dim = 9;

%% Projection

octaMat = LoadOctaMatsScaled;
octaMat = cell2mat(permute(octaMat, [2 3 1]));
SdpA = cat(3, blkdiag(1, zeros(9)), octaMat);
SdpA = reshape(SdpA, [10^2, 16])';
SdpB = [1; zeros(15, 1)];

octahedral.proj = @(q) MultiSdp(q, SdpA, SdpB);

%% Boundary Alignment

octahedral.projAligned = @(q) sqrt(5/12) * (q ./ vecnorm(q, 2, 1));

octahedral.bdryBasis = @octaBdryBasis;
function [bdryFixed, bdryBasis] = octaBdryBasis(bdryNormals)
    BdryRotStacked = OctaAlignMat(bdryNormals);
    bdryBasis = multiprod(multitransp(BdryRotStacked), sparse([1 9], [1 2], [1, 1], 9, 2));
    bdryFixed = multiprod(multitransp(BdryRotStacked), [0 0 0 0 sqrt(7/12) 0 0 0 0]', [1 2], 1);
end

%% Initial value

octahedral.rand = @RandOctahedralField;

end
