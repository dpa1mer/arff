function q = ExpSO3(axisAngles, q, YZ, rotateNorthOnly)
% Implements the exponential map in representations of SO(3).
% Multiplies each column of q by the exponential of the
% Lie algebra element in the corresponding row of axisAngles.

bandIdx = floor(size(YZ, 1) / 2);
bands = (-bandIdx:bandIdx).';

function q = expLz(ang, band, q, qr)
    q = cos(ang * band) * q - sin(ang * band) * qr;
end
function q = expLzT(ang, band, q, qr)
    q = cos(ang * band) * q + sin(ang * band) * qr;
end

%% Extract angles

[az, el, rot] = cart2sph(axisAngles(:, 1), axisAngles(:, 2), axisAngles(:, 3));
preservedIdx = (rot == 0);
qPreserved = q(:, preservedIdx);
el = pi/2 - el;

gpuflag = isa(q, 'gpuArray');
if ~gpuflag
    azAngs = bands .* az.';
    elAngs = bands .* el.';
    rotAngs = bands .* rot.';
    cosAz = cos(azAngs);
    sinAz = sin(azAngs);
    cosEl = cos(elAngs);
    sinEl = sin(elAngs);
    cosRot = cos(rotAngs);
    sinRot = sin(rotAngs);
end

%% Rotate axis to [0, 0, 1]'

if gpuflag
    q = arrayfun(@expLzT, az.', bands, q, flipud(q));
else
    q = cosAz .* q + sinAz .* flipud(q);
end
q = YZ * q;
if gpuflag
    q = arrayfun(@expLzT, el.', bands, q, flipud(q));
else
    q = cosEl .* q + sinEl .* flipud(q);
end
q = YZ' * q;

if nargin < 4 || ~rotateNorthOnly
    %% Rotate around [0, 0, 1]'

    if gpuflag
        q = arrayfun(@expLz, rot.', bands, q, flipud(q));
    else
        q = cosRot .* q - sinRot .* flipud(q);
    end

    %% Rotate [0, 0, 1]' back to axis

    q = YZ * q;
    if gpuflag
        q = arrayfun(@expLz, el.', bands, q, flipud(q));
    else
        q = cosEl .* q - sinEl .* flipud(q);
    end
    q = YZ' * q;
    if gpuflag
        q = arrayfun(@expLz, az.', bands, q, flipud(q));
    else
        q = cosAz .* q - sinAz .* flipud(q);
    end

    q(:, preservedIdx) = qPreserved;
end

end

