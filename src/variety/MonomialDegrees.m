function degree = MonomialDegrees(d)

index = find(hankel(d+1:-1:1));
deg_x = repmat((0:d)', 1, d+1);
deg_y = repmat(0:d, d+1, 1);
deg_z = hankel(d:-1:0);
degree = [deg_x(index), deg_y(index), deg_z(index)]';

end