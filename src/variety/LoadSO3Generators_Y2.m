function [Lx, Ly, Lz, YZ] = LoadSO3Generators_Y2

persistent cachedLx cachedLy cachedLz cachedYZ;
if isempty(cachedLx)
    cachedLx = [0, 0, 0,       -1,  0;
                0, 0, -sqrt(3), 0, -1;
                0, sqrt(3), 0,  0,  0;
                1, 0,       0,  0,  0;
                0, 1,       0,  0,  0];

    cachedLy = [0,  1, 0,        0,  0;
                -1, 0, 0,        0,  0;
                0,  0, 0, -sqrt(3),  0;
                0,  0, sqrt(3),  0, -1;
                0,  0, 0,        1,  0];

    cachedLz = [0,  0, 0, 0, 2;
                0,  0, 0, 1, 0;
                0,  0, 0, 0, 0;
                0, -1, 0, 0, 0;
                -2, 0, 0, 0, 0];

    cachedYZ = double(expm((pi/2) * sym(cachedLx)));
end

Lx = cachedLx;
Ly = cachedLy;
Lz = cachedLz;
YZ = cachedYZ;

end