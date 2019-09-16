function ExportMesh(filename, meshData)

vertTable = [array2table(["MeshVersionFormatted", "1", "", "";
                 "Dimension", "3", "", "";
                 "Vertices", meshData.nv, "", ""]);
             array2table([meshData.verts zeros(size(meshData.verts, 1), 1)])];
writetable(vertTable, filename, 'FileType', 'text', 'Delimiter', ' ', 'WriteVariableNames', false);

nt = size(meshData.tets, 1);
tetTable = [array2table(["Tetrahedra", nt, "", "", ""]);
            array2table([meshData.tets zeros(nt, 1)])];

tempFilename = [filename '.temp'];
writetable(tetTable, tempFilename, 'FileType', 'text', 'Delimiter', ' ', 'WriteVariableNames', false);
system(['cat ' tempFilename ' >> ' filename]);
system(['rm ' tempFilename]);

end