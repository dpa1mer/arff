q0 = randn(9, 100000);
q0 = q0 ./ vecnorm(q0, 2, 1);
tic;
qRay = RayProjection(q0);
tRay = toc;

octa = OctaMBO;
tic;
qOurs = octa.proj(q0);
tOurs = toc;
dOurs = vecnorm(qOurs - q0, 2, 1);
dRay = vecnorm(qRay - q0, 2, 1);
[~, idx] = sort(dRay - dOurs, 'descend');

for k = 1:5
    fOurs = Octa2Frames(qOurs(:, idx(k)));
    fOurs = 2 * [fOurs, -fOurs];
    fRay = Octa2Frames(qRay(:,idx(k)));
    fRay = 2 * [fRay, -fRay];
    figure; PlotRealY4(q0(:, idx(k)), sqrt(189/pi)/8, 1, [0 0 0], 500); view(3);
    axis image vis3d off;
    hold on;
    lighting gouraud;
    camlight; camlight headlight;
    shading interp;
    quiver3(zeros(6, 1), zeros(6, 1), zeros(6, 1), fOurs(1, :)', fOurs(2, :)', fOurs(3, :)', ...
        'b', 'ShowArrowHead', 'off', 'LineWidth', 3);
    quiver3(zeros(6, 1), zeros(6, 1), zeros(6, 1), fRay(1, :)', fRay(2, :)', fRay(3, :)', ...
        'r', 'ShowArrowHead', 'off', 'LineWidth', 3);

    h = coneplot(0.9 * fOurs(1, :)', 0.9 * fOurs(2, :)', 0.9 * fOurs(3, :)', fOurs(1, :)', fOurs(2, :)', fOurs(3, :)', 0.05, 'nointerp');
    h.FaceColor = 'b'; h.EdgeColor = 'none';
    
    h = coneplot(0.9 * fRay(1, :)', 0.9 * fRay(2, :)', 0.9 * fRay(3, :)', fRay(1, :)', fRay(2, :)', fRay(3, :)', 0.05, 'nointerp');
    h.FaceColor = 'r'; h.EdgeColor = 'none';
    
    set(gcf, 'color', 'white');
end