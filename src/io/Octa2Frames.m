function frames = Octa2Frames(q)
n = size(q, 2);

q = [(sqrt(189)/4) * ones(1, n); q ./ vecnorm(q, 2, 1)];

v1 = randn(n, 3);
v2 = randn(n, 3);
delta = 1;

while delta > eps^0.9 * sqrt(n)
    x = v1(:, 1);
    y = v1(:, 2);
    z = v1(:, 3);

    w1 = OctaTensorGradient(q, x, y, z);
    w2 = OctaTensorGradient(q, v2(:, 1), v2(:, 2), v2(:, 3));

    v1 = w1 ./ vecnorm(w1, 2, 2);
    v2 = w2 - dot(w2, v1, 2) .* v1;
    v2 = v2 ./ vecnorm(v2, 2, 2);
    delta = norm(v1 - [x, y, z], 'fro');
end

v3 = cross(v1, v2, 2);

frames = [permute(v1, [2 3 1]) permute(v2, [2 3 1]) permute(v3, [2 3 1])];

end
