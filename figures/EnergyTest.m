function info = EnergyTest

nSamples = 10;

datadir = '../meshes/spheres/';
datafiles = dir(fullfile(datadir, 'sphere*'));
for item = 1 : length(datafiles)
    s = ImportMesh(fullfile(datadir, datafiles(item).name));
    vols = TetVolumes(s.verts, s.tets);
    info(item).volGeomean = geomean(vols);
    info(item).volMean = mean(vols);
    info(item).volMax = max(vols);
    info(item).volMin = min(vols);
    info(item).numTets = length(s.tets);
    info(item).numVerts = s.nv;
    
    % Octahedral
    for j = 1:nSamples
        qOcta = OctaManopt(s, []);
        qOcta = Octa2Odeco(qOcta);
        info(item).EOcta(j) = 0.5 * sum(dot(qOcta.', s.L * qOcta.'));


        % Odeco
        qOdeco = OdecoManopt(s, [], false, true);
        info(item).EOdeco(j) = 0.5 * sum(dot(qOdeco.', s.L * qOdeco.'));
    end
end

vols = repelem([info.volGeomean], nSamples);

figure; scatter(vols, [info.EOcta], '.');
hold on; scatter(vols, [info.EOdeco], '.');
set(gca, 'XScale', 'log');
set(gca, 'Xdir', 'reverse');

end
