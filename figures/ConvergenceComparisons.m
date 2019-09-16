function ConvergenceComparisons(inputdir, outputdir)

%% Run both methods on all models

cond = {'raycond', 'ourcond'};
calls{1}.ray = @(meshData, ~) Ray(meshData, [], false, false);
calls{1}.rtr = @(meshData) OctaManopt(meshData, RayInit(meshData), false, false, true);
calls{2}.ray = @(meshData, q0) Ray(meshData, q0, true, false);
calls{2}.rtr = @(meshData) OctaManopt(meshData, [], false, false);

meshfiles = dir(fullfile(inputdir, '*.mesh'));
for i = 1:length(meshfiles)
    [~, basename, ext] = fileparts(meshfiles(i).name);
    fullname = fullfile(meshfiles(i).folder, [basename ext]);
    fprintf('fullname %s\n', fullname);

    meshData = ImportMesh(fullname);
    
    for j = 1:2
        ray = calls{j}.ray;
        rtr = calls{j}.rtr;
        [~, q0, infoRTR] = rtr(meshData);
        [~, ~, infoRay] = ray(meshData, q0);
        
        gradRTR = (20/3) * [infoRTR.gradnorm];
        timeRTR = [infoRTR.time];
        
        gradRay = [infoRay.gradnorm];
        timeRay = [infoRay.time];
        
        figure; semilogy(timeRay, gradRay); hold on; semilogy(timeRTR, gradRTR);
        
        writematrix([timeRay.' gradRay.'], fullfile(outputdir, [basename '_' cond{j} '_grad_ray.csv']), 'FileType', 'text');
        writematrix([timeRTR.' gradRTR.'], fullfile(outputdir, [basename '_' cond{j} '_grad_rtr.csv']), 'FileType', 'text');
    end
end

end