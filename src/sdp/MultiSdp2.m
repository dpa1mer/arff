function solver = MultiSdp2(A, b)
%MULTISDP2 Solves many small SDPs with common equality constraints
% coming from Euclidean projection relaxations.
% Based on [Helmberg et al. 1996], [Yamashita et al. 2002], etc.

eps_primal = 1e-7;
eps_gap = 1e-7;

d = size(A, 1);
assert(d == size(A, 2));
k = size(A, 3);

gpuflag = false;
dataType = 'double';
if gpuDeviceCount > 0
    % Set up data on GPU
    gd = gpuDevice;
    gpuflag = true;
    dataType = 'gpuArray';
    maxGpuAllocSize = floor(gd.TotalMemory / (8 * 8));
    A = gpuArray(A);
    b = gpuArray(b);
end
Id = eye(d, d, dataType);

% Rescale A and b
normA = sqrt(sum(sum(A.^2)));
A = A ./ normA;
b = b ./ squeeze(normA);

% Reshape A in various forms
At = sparse(reshape(A, d^2, k));
Aflat = At.';
Astacked = reshape(A, d, d * k).';
AoI = sparse(kron(Astacked, Id));
IoA = zeros(d ^2, d^2, k, dataType);
for j = 1 : k
    IoA(:, :, j) = kron(Id, A(:, :, j));
end
IoA = sparse(reshape(IoA, d^2, d^2 * k).');

solver.project = @sdpProject;
function [X_final, Z_final, unconvergedIdx, gap, feasPrimal] = sdpProject(q0, maxiter, pred_corr)
    if nargin < 3 || ~pred_corr
        pred_corr = 1;
    else
        pred_corr = 2;
    end
    
    n = size(q0, 2);
    if gpuflag
        q0 = gpuArray(q0);
    end
    
    % Put the problem in maximization form
    q0 = permute(q0, [1 3 2]);
    C = [-2 * sum(q0.^2, 1), multitransp(q0);
         q0, repmat(-eye(d - 1, d - 1, dataType), [1 1 n])];
    C = C ./ sqrt(sum(sum(C.^2)));

    %% Initial values: Primal X, Dual y, Z
    X = repmat(Id, [1 1 n]);     % Set X to be positive definite
    y = zeros(k, n, dataType); % Set y to 0 so that Z will be positive definite
    Aty = reshape(At * y, d, d, n);
    Z = Aty - C;
    
    %% Final values: initialize to zero
    X_final = zeros(d, d, n, dataType);
    Z_final = zeros(d, d, n, dataType);
    if gpuflag
        unconvergedIdx = gpuArray.colon(1, n);
    else
        unconvergedIdx = 1:n;
    end

    for iter = 1:maxiter

        %% Select unconverged pages
        feasPrimal = vecnorm(Aflat * reshape(X, d^2, n) - b, 2, 1).';
        gap = sum(b .* y, 1).' - squeeze(sum(dot(C, X)));
        feasibleMask = feasPrimal < eps_primal;
        convergedMask = feasibleMask & abs(gap) < eps_gap;
        if any(convergedMask)
            X_final(:, :, unconvergedIdx(convergedMask)) = X(:, :, convergedMask);
            Z_final(:, :, unconvergedIdx(convergedMask)) = Z(:, :, convergedMask);
            unconvergedMask = ~convergedMask;
            nUnconverged = sum(unconvergedMask);
            if nUnconverged == 0
                unconvergedIdx = [];
                feasPrimal = [];
                gap = [];
                break;
            else
                unconvergedIdx = unconvergedIdx(unconvergedMask);
                X = X(:, :, unconvergedMask);
                Z = Z(:, :, unconvergedMask);
                C = C(:, :, unconvergedMask);
                G = G(:, :, unconvergedMask);
                y = y(:, unconvergedMask);
                feasibleMask = feasibleMask(unconvergedMask);
                n = nUnconverged;
            end
        end

        %% Auxiliary computations
        Aty = reshape(At * y, d, d, n);
        F_dual = Z + C - Aty;
        F_dual_X = batchop('mult', F_dual, X);

        ZX = batchop('mult', Z, X);
        ZdotX = sum(dot(Z, X));
        LX = batchop('chol', X); % Upper triangle is garbage
        LZ = batchop('chol', Z); % Upper triangle is garbage
        Zinv = batchop('cholsolve', LZ, repmat(Id, [1 1 n]));
        
        %% Form System Matrix: G_ij = G_ji = trace(A_i * Zinv * A_j * X)
        
        if gpuflag % Break into smaller blocks to avoid memory pressure
            nBlks = ceil(2 * d^2 * k * n / maxGpuAllocSize);
            blkMax = ceil(n / nBlks);
            for blk = 1:nBlks
                blkStart = (blk - 1) * blkMax + 1;
                blkEnd = min(blk * blkMax, n);
                blkSize = blkEnd - blkStart + 1;
                ZinvA = reshape(AoI * reshape(Zinv(:, :, blkStart : blkEnd), d^2, blkSize), d^2, k, blkSize);
                AX = reshape(IoA * reshape(X(:, :, blkStart : blkEnd), d^2, blkSize), d^2, k, blkSize);
                G(:, :, blkStart : blkEnd) = batchop('mult', AX, ZinvA, 'T', 'N');
            end
        else
            ZinvA = reshape(AoI * reshape(Zinv, d^2, n), d^2, k, n);
            AX = reshape(IoA * reshape(X, d^2, n), d^2, k, n);
            G = batchop('mult', AX, ZinvA, 'T', 'N');
        end
        LG = batchop('chol', G);

        %% Mehrotra Predictor-Corrector
        
        sigma = 0.5 * ones(1, 1, n, dataType);
        sigma(feasibleMask) = 0;
        for substep = 1:pred_corr
            mu = ZdotX ./ d;
            mu = sigma .* mu;

            % Solve for Search Directions
            F_ZX = ZX - mu .* Id;
            if substep == 2
                F_ZX = F_ZX + batchop('mult', DZ, DX);
            end

            DyRhs = -b + permute(Aflat * reshape(batchop('mult', Zinv, F_dual_X - F_ZX) + X, d^2, n), [1 3 2]);
            Dy = squeeze(batchop('cholsolve', LG, DyRhs));
            AtDy = reshape(At * Dy, d, d, n);
            DZ = -F_dual + AtDy;
            DX = multisym(batchop('mult', Zinv, -F_ZX - batchop('mult', DZ, X)));

            % Choose Step Size
            LDXL = batchop('cholcong', LX, DX);
            LDZL = batchop('cholcong', LZ, DZ);
            diagLDXL = Id .* LDXL;
            diagLDZL = Id .* LDZL;
            ub_DX = max(sum(-diagLDXL + abs(-LDXL + diagLDXL), 2), [], 1);
            ub_DZ = max(sum(-diagLDZL + abs(-LDZL + diagLDZL), 2), [], 1);
            ub_DX = min(ub_DX, sqrt(sum(sum(LDXL.^2))));
            ub_DZ = min(ub_DZ, sqrt(sum(sum(LDZL.^2))));

            a_primal = min(0.98 * ub_DX.^(-1), 1);
            a_dual = min(0.98 * ub_DZ.^(-1), 1);

            % Centering parameter for corrector step
            if substep == 1
                sigma = (sum(dot(Z + a_dual .* DZ, X + a_primal .* DX)) ./ ZdotX).^3;
                sigma = max(min(sigma, 1), 0.2);
            end
        end

        y = y + reshape(a_dual, [1 n]) .* Dy;
        Z = Z + a_dual .* DZ;
        X = X + a_primal .* DX;
    end

end

end

