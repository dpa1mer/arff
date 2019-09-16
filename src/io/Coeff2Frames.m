function frames = Coeff2Frames(q, normalize)
if nargin < 2
    normalize = false;
end

if size(q, 1) == 9 % Octahedral
    frames = Octa2Frames(q);
else % Odeco
    frames = Odeco2Frames(Sph024ToMonomial(q));
    if normalize
        frames = frames ./ vecnorm(frames, 2, 1);
    end
end

end