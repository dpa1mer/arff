function [succeeded, results] = GenerateComparisons(inputdir, outputdir)

%% List of Methods

types(1).name = 'Octa';
types(1).fiber = OctaMBO;
types(1).rtr = @(meshData, q0) OctaManopt(meshData, q0, false, false);
types(2).name = 'Odeco';
types(2).fiber = OdecoMBO;
types(2).rtr = @(meshData, q0) OdecoManopt(meshData, q0, false, false);

c = 1;
config(c).type = 1;
config(c).method{1} = 'Ray';
config(c).run{1} = @(meshData, ~) Ray(meshData);

c = c + 1;
config(c).type = 1;
config(c).method{1} = 'Ray2';
config(c).run{1} = @(meshData, q0) Ray(meshData, q0, true);

c = c + 1;
config(c).type = 1;
config(c).method{1} = 'RayMBO';
config(c).run{1} = @(meshData, q0) MBO(meshData, RayMBO, q0);

c = c + 1;
config(c).type = 1;
config(c).method{1} = 'RayMMBO';
config(c).run{1} = @(meshData, q0) MBO(meshData, RayMBO, q0, 50, 3);


for t = 1:length(types)
    c = c + 1;
    config(c).type = t;
    config(c).method{1} = 'RTR';
    config(c).run{1} = types(t).rtr;

    c = c + 1;
    config(c).type = t;
    config(c).method{1} = 'MBO';
    config(c).run{1} = @(meshData, q0) MBO(meshData, types(t).fiber, q0);

    c = c + 1;
    config(c).type = t;
    config(c).method{1} = 'mMBO';
    config(c).run{1} = @(meshData, q0) MBO(meshData, types(t).fiber, q0, 50, 3);
end

for c = 1:length(config)
    if ~strcmp(config(c).method{1}, 'RTR')
        config(c).method{2} = [config(c).method{1} '+RTR'];
        config(c).run{2} = types(config(c).type).rtr;
    end
end

%% Run all methods on all models

% counter for result table
trial = 0;
meshfiles = dir(fullfile(inputdir, '*.mesh'));
for i = 1:length(meshfiles)
    [~, basename, ext] = fileparts(meshfiles(i).name);
    fullname = fullfile(meshfiles(i).folder, [basename ext]);
    fprintf('fullname %s\n', fullname);

    meshData = ImportMesh(fullname);
    q0_Octa = RandOctahedralField(meshData.nv, meshData.bdryIdx, meshData.bdryNormals);
    q0_Odeco = Octa2Odeco(q0_Octa);
    for j = 1:length(config)
        totalTime = 0;
        totalIters = 0;
        for k = 1:length(config(j).method)
            trial = trial + 1;
            trialName = [basename '_' types(config(j).type).name '_' config(j).method{k}];
            try
                run = config(j).run{k};
                if k > 1
                    [q, q0, info] = run(meshData, q);
                elseif config(j).type == 1 
                    [q, q0, info] = run(meshData, q0_Octa);
                else
                    [q, q0, info] = run(meshData, q0_Odeco);
                end

                % Compute energy consistently for all types and methods
                qOdeco = q;
                if size(q, 1) == 9
                    qOdeco = Octa2Odeco(q);
                end
                energy = 0.5 * sum(dot(qOdeco.', meshData.L * qOdeco.'));

                totalIters = totalIters + length(info);
                totalTime = totalTime + info(end).time;

                results(trial).Model = basename;
                results(trial).Vertices = meshData.nv;
                results(trial).Type = types(config(j).type).name;
                results(trial).Method = config(j).method{k};
                results(trial).Energy = energy;
                results(trial).Time = totalTime;
                results(trial).Iterations = totalIters;

                % save complete matlab output
                ExportFrames(fullfile(outputdir, [trialName '_frames.txt']), Coeff2Frames(q));
                ExportFrames(fullfile(outputdir, [trialName '_init_frames.txt']), Coeff2Frames(q0));
                save(fullfile(outputdir, [trialName '.mat']), 'trialName', 'q', 'q0', 'info');
                succeeded(trial) = true;
            catch exception
                warning([trialName ' GenerateComparisons: ' exception.message]);
                succeeded(trial) = false;
                break;
            end
        end
    end
end

% store result table
save(fullfile(outputdir, 'results.mat'), 'results');
resultTable = struct2table(results);
writetable(resultTable, fullfile(outputdir, 'results.csv'), 'FileType', 'text');

end