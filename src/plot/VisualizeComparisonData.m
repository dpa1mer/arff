function q = VisualizeComparisonData(meshData)

[filepath, filename, ~] = fileparts(meshData.file);
rayFrames = ImportFrames(fullfile(filepath, [filename '_frame.txt']));
q = Frames2Octa(rayFrames);

VisualizeFrameField(rayFrames, dot(q.', meshData.L * q.', 2), meshData.tetra, meshData.bdry);

end

