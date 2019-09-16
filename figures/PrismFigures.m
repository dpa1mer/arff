function PrismFigures

prism = ImportMesh('../meshes/prism/prism.node', pi/2 + 1e-2);

qPrismOdecoMBO = MBO(prism, OdecoMBO, [], 1, 0);
qPrismOdecoMBORTR = OdecoManopt(prism, qPrismOdecoMBO);

samples = RadialSamples(6, 10, pi/2) ./ 8 + [0 sqrt(3)/3 0];
samplesStacked = [samples + [0 0 1]; samples; samples + [0 0 -1]];
samplesLine = [zeros(14, 1) linspace(0, sqrt(3), 14)' zeros(14, 1)];

figure; PlotInterpolatedFrames(qPrismOdecoMBORTR, prism.tetra, samplesStacked, 'ColorField', @(q) vecnorm(q(2:6, :), 2, 1).');
view([0 -1 0.5]);
axis image vis3d off;
caxis('manual');
hold on; trisurf(prism.bdry, 'FaceColor', 'black', 'EdgeColor', 'none', 'FaceAlpha', 0.02);
camproj('perspective');
set(gcf, 'color', 'white');
camlight;
camlight headlight;

shadow = polybuffer([0 0; 0 sqrt(3)], 'lines', 0.06);
shadowColor = [0.8 0.8 0.8];
shadV = [shadow.Vertices repmat(-0.04, [size(shadow.Vertices, 1) 1])];

[dotx, doty, dotz] = sphere(100);
dotx = 0.01 * dotx;
doty = 0.01 * doty + sqrt(3)/3;
dotz = 0.01 * dotz;

randRot = eul2rotm(2 * pi * rand(1, 3));

figure; PlotInterpolatedFrames(qPrismOdecoMBORTR, prism.tetra, samples, 'NormalColorRot', randRot);
hold on; patch(shadV(:, 1), shadV(:, 2), shadV(:, 3), shadowColor, 'EdgeColor', 'none');
hold on; surf(dotx, doty, dotz, 'EdgeColor', 'none', 'FaceColor', 'k');
view([0 -1 2]);
axis image vis3d off;
camproj('perspective');
set(gcf, 'color', 'white');

figure; PlotInterpolatedFrames(qPrismOdecoMBORTR, prism.tetra, samplesLine, 'NormalColorRot', randRot);
hold on; patch(shadV(:, 1), shadV(:, 2), shadV(:, 3), shadowColor, 'EdgeColor', 'none');
hold on; surf(dotx, doty, dotz, 'EdgeColor', 'none', 'FaceColor', 'k');
view([-3 0 1]);
axis image vis3d off;
camproj('perspective');
set(gcf, 'color', 'white');

end