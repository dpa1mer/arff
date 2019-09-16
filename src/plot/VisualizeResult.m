function VisualizeResult(meshData, q)

frames = Coeff2Frames(q);
if size(q, 1) == 9 % Octahedral
    energy = dot(q.', meshData.L * q.', 2);
else % Odeco
    energy = dot(q(7:15, :).', meshData.L * q(7:15, :).', 2);
end
VisualizeFrameField(frames, energy, meshData.tetra, meshData.bdry);

end

