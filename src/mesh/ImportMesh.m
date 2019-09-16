function meshData = ImportMesh(meshFile, bdryAngleCutoff)

if nargin < 2
    bdryAngleCutoff = 0;
end

[filepath, filename, ext] = fileparts(meshFile);

if strcmp(ext, '.mesh')
    fid = fopen(meshFile,'r');
    textscan(fid, 'MeshVersionFormatted %d');
    textscan(fid, 'Dimension %d', 'Whitespace', ' \t\n', 'MultipleDelimsAsOne', true);
    
    textscan(fid, 'Vertices %d', 'Whitespace', ' \t\n', 'MultipleDelimsAsOne', true);
    verts = textscan(fid, '%f %f %f %*d', 'CollectOutput', true, 'MultipleDelimsAsOne', true);
    
    textscan(fid, 'Triangles %d', 'Whitespace', ' \t\n');
    textscan(fid, '%d %d %d %*d', 'CollectOutput', true, 'MultipleDelimsAsOne', true);
    
    textscan(fid, 'Edges %d', 'Whitespace', ' \t\n');
    textscan(fid, '%d %d %*d', 'CollectOutput', true, 'MultipleDelimsAsOne', true);
    
    textscan(fid, 'Tetrahedra %d', 'Whitespace', ' \t\n');
    tets = textscan(fid, '%d %d %d %d %*d', 'CollectOutput', true, 'MultipleDelimsAsOne', true);
    
    fclose(fid);

    verts = verts{:};
    tets = double(tets{:});
    
elseif strcmp(ext, '.node') || strcmp(ext, '.ele')
    nodeFile = fullfile(filepath, [filename '.node']);
    fid = fopen(nodeFile, 'r');
    verts = textscan(fid, '%d %f %f %f', 'HeaderLines', 1, 'CollectOutput', true);
    fclose(fid);
    
    idx = verts{1};
    verts = verts{2};
    
    eleFile = fullfile(filepath, [filename '.ele']);
    fid = fopen(eleFile, 'r');
    nTets = textscan(fid, '%d %*d %*d', 1, 'CollectOutput', true);
    tets = textscan(fid, '%*d %d %d %d %d %*s', nTets{:}, 'HeaderLines', 1, 'CollectOutput', true);
    fclose(fid);
    
    tets = double(tets{:});
    if (idx(1) == 0)
        tets = tets + 1;
    end
else
    error('Unknown file type %s', ext);
end

meshData = ProcessMesh(verts, tets, bdryAngleCutoff);
meshData.file = meshFile;

end