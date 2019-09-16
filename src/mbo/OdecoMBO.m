function odeco = OdecoMBO

odeco.dim = 15;

%% Projection

odecoMat = LoadOdecoMatsSph;
odecoMat = cell2mat(permute(odecoMat, [2 3 1]));
SdpA = cat(3, blkdiag(1, zeros(15)), ...
              [zeros(1, 16, 27); zeros(15, 1, 27) odecoMat]);
SdpA = reshape(SdpA, [16^2, 28])';
SdpB = [1; zeros(27, 1)];
odeco.proj = @(q0) MultiSdp(q0, SdpA, SdpB);


%% Boundary Alignment

AlignedSdpA = cat(3, diag([1 0 0 -18 -18]), ...
                     diag([sqrt(2) -6 -6], 2), ...
                     diag([sqrt(2) 0 -6 0], 1) + diag([0 6], 3));
AlignedSdpA = cat(3, blkdiag(1, zeros(5)), blkdiag(0, eye(5)), ...
                     [zeros(1, 6, 3); zeros(5, 1, 3) AlignedSdpA]);
AlignedSdpA = reshape(AlignedSdpA, [6^2, 5])';
AlignedSdpB = [1; 1; 0; 0; 0];
odeco.projAligned = @odecoAlignedProj;
function q = odecoAlignedProj(q0)
    q = MultiSdp(q0, AlignedSdpA, AlignedSdpB);
    
    % Rescale q (overall problem is min |q0 - q|^2)
    q = (315/(64*pi)) * dot(q0, q, 1) .* q;
end

zAlignedBasis = ...
    [(4/15).*(2.*pi).^(1/2),0,0,0,0;
    0,(-4/7).*((3/5).*pi).^(1/2),0,0,0;
    0,0,0,0,0;
    (-8/21).*((2/5).*pi).^(1/2),0,0,0,0;
    0,0,0,0,0;
    0,0,(4/7).*((3/5).*pi).^(1/2),0,0;
    0,0,0,(-8/3).*((1/35).*pi).^(1/2),0;
    0,0,0,0,0;
    0,(4/21).*((1/5).*pi).^(1/2),0,0,0;
    0,0,0,0,0;
    (4/105).*(2.*pi).^(1/2),0,0,0,0;
    0,0,0,0,0;
    0,0,(-4/21).*((1/5).*pi).^(1/2),0,0;
    0,0,0,0,0;
    0,0,0,0,(8/3).*((1/35).*pi).^(1/2)];
zAlignedFixed = [(2/5).*pi.^(1/2),0,0,(8/7).*((1/5).*pi).^(1/2),0,0,0,0,0,0,(16/105).*pi.^(1/2),0,0,0,0]';
odeco.bdryBasis = @odecoBdryBasis;
function [bdryFixed, bdryBasis] = odecoBdryBasis(bdryNormals)
    BdryRotStacked = OdecoAlignMat(bdryNormals);
    bdryBasis = multiprod(multitransp(BdryRotStacked), zAlignedBasis);
    bdryFixed = multiprod(multitransp(BdryRotStacked), zAlignedFixed, [1 2], 1);
end

%% Initial Value

odeco.rand = @(nv, bdryIdx, bdryNormals) Octa2Odeco(RandOctahedralField(nv, bdryIdx, bdryNormals));

end
