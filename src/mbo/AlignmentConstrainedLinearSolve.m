function [q, iter, soln] = AlignmentConstrainedLinearSolve(A, rhs, intIdx, bdryIdx, BdryBasis, qFixed, warmstart)

gpuflag = false;
if gpuDeviceCount > 0
    gpuflag = true;
    A = gpuArray(A);
    rhs = gpuArray(rhs);
    bdryIdx = gpuArray(bdryIdx);
    BdryBasis = gpuArray(BdryBasis);
    qFixed = gpuArray(qFixed);
    warmstart = gpuArray(warmstart);
end

D = size(BdryBasis, 1);
d = size(BdryBasis, 2);
nb = length(bdryIdx);
ni = length(intIdx);
bdryIntIdx = [bdryIdx; intIdx];

function Aq = mulA(q)
    qBdry = reshape(q(1:d*nb), [d nb]);
    qBdry = multiprod(BdryBasis, qBdry, [1 2], 1);
    qFull(:, bdryIntIdx) = [qBdry, reshape(q(d*nb+1:end), [D ni])];
    AqFull = (A * qFull.').';
    AqBdry = multiprod(multitransp(BdryBasis), AqFull(:, bdryIdx), [1 2], 1);
    AqBdry = AqBdry(:);
    Aq = [AqBdry; reshape(AqFull(:, intIdx), [D*ni 1])];
end

rhs = rhs - (A * qFixed.').';
rhsBdry = multiprod(multitransp(BdryBasis), rhs(:, bdryIdx), [1 2], 1);
rhs = [rhsBdry(:); reshape(rhs(:, intIdx), [D*ni, 1])];
[soln, ~, ~, iter] = pcg(@mulA, rhs, [], 1000, [], [], warmstart);
q(:, bdryIdx) = multiprod(BdryBasis, reshape(soln(1:d*nb), [d nb]), [1 2], 1);
q(:, intIdx) = reshape(soln(d*nb+1:end), [D ni]);
q = q + qFixed;

if gpuflag
    q = gather(q);
    soln = gather(soln);
end

end
