function vol = TetVolumes(verts, tets)
v0 = verts(tets(:, 1), :);
v1 = verts(tets(:, 2), :);
v2 = verts(tets(:, 3), :);
v3 = verts(tets(:, 4), :);

vol = abs(dot(v1 - v0, cross(v2 - v0, v3 - v0, 2), 2)) / 6;
end

