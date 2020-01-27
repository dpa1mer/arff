function octahedral = RayMBO

octahedral.dim = 9;

%% Ray's Projection

octahedral.proj = @(q) RayProjection(q);

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
