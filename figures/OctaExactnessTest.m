function [worstEig2, eigRatio, q0, q, Q] = OctaExactnessTest(n)

q0 = randn(9, n);
q0 = q0 ./ vecnorm(q0, 2, 1);

octaMat = LoadOctaMatsScaled;
octaMat = cell2mat(permute(octaMat, [2 3 1]));
SdpA = cat(3, blkdiag(1, zeros(9)), octaMat);
SdpA = reshape(SdpA, [10^2, 16])';
SdpB = [1; zeros(15, 1)];

[q, Q] = MultiSdp(q0, SdpA, SdpB);

Q = reshape(Q, 10, 10, n);
firstEig = zeros(n, 1);
secondEig = zeros(n, 1);
for i = 1:n
    ev = eig(Q(:, :, i));
    firstEig(i) = ev(10);
    secondEig(i) = ev(9);
end

eigRatio = secondEig ./ firstEig;
figure; histogram(log10(eigRatio), 'LineStyle', 'none');
worstEig2 = max(secondEig);

end