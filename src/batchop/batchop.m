function varargout = batchop(op, A, varargin)
%BATCHOP - Batched operations on 3D matrices
%    BATCHOP performs an identical operation on each "page" of a double
%    precision array, either on CPU or GPU.
%
%    C = BATCHOP('mult', A, B, 'N' or 'T', 'N' or 'T') performs matrix
%    multiplication on each page. For example,
%        C = BATCHOP('mult', A, B, 'N', 'T')
%    computes C(:, :, k) = A(:, :, k) * B(:, :, k)' for each k.
%
%    X = BATCHOP('leastsq', A, B) solves the least-squares (overdetermined)
%    system A(:, :, k) * X(:, :, k) = B(:, :, k) for each page.
%
%    [L, info] = BATCHOP('chol', A) returns the left Cholesky factor in the
%    lower triangle of each L(:, :, k) for which info(k) == 0. Note: the upper
%    triangle of each page is not necessarily set to zero.
%
%    X = BATCHOP('cholsolve', L, B) solves the equation
%        A(:, :, k) * X(:, :, k) = B(:, :, k)
%    for each k, where L is the left Cholesky factor of A.
%
%    X = BATCHOP('cholcong', L, B) computes the congruence
%        X(:, :, k) = L(:, :, k) \ B(:, :, k) / L(:, :, k)'
%    for each k.
%
%    [U, S, V, info] = BATCHOP('svd', A, rank) performs partial SVD on each
%    page:
%        A(:, :, k) = U(:, :, k) * (S(:, k) .* V(:, :, k))    (CPU)
%        A(:, :, k) = U(:, :, k) * (S(:, k) .* V(:, :, k)')   (GPU)
%
%    B = BATCHOP('pinv', A, rank) computes the pseudoinverse of each page of
%    A, assuming the given rank.
%    
%On CPU only:
%    [Q, R, p, info] = BATCHOP('qr', A) performs QR decomposition with pivoting
%    for each page: A(:, p, k) = Q(:, :, k) * R(:, :, k).
%
%    X = BATCHOP('trisolve', T, B, 'L' or 'U') solves the lower- or
%    upper-triangular system T(:, :, k) * X(:, :, k) = B(:, :, k) for each k.
%
%On GPU only:
%    [E, Q, info] = BATCHOP('eig', A)
%    [E, info] = BATCHOP('eigval', A) (eigenvalues only.)
%
%See also PAGEFUN, ARRAYFUN, MULTIPROD, MULTITRANSP


gpuflag = isa(A, 'gpuArray');

if strcmp(op, 'pinv')
    rank = varargin{1};
    if gpuflag
        [U, S, V] = batchop_gpu('svd', A, rank);
        S = permute(S, [3 1 2]);
        varargout{1} = batchop_gpu('mult', V ./ S, U, 'N', 'T');
    else
        [U, S, Vt] = batchop_cpu('svd', A, rank);
        S = permute(S, [1 3 2]);
        varargout{1} = batchop_cpu('mult', Vt ./ S, U, 'T', 'T');
    end
elseif gpuflag
    [varargout{1:nargout}] = batchop_gpu(op, A, varargin{:});
else
    [varargout{1:nargout}] = batchop_cpu(op, A, varargin{:});
end

end