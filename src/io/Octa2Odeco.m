function q = Octa2Odeco(q)

n = size(q, 2);
q = [((6/5)*sqrt(pi)) * ones(1, n); zeros(5, n); ((8/5)*sqrt(pi/21)) * q];

end

