function f = Sph024ToMonomial(q)

f = [12.*pi.^(-1/2),0,4.*pi.^(-1/2),0,12.*pi.^(-1/2),0,0,0,0,4.* ...
pi.^(-1/2),0,4.*pi.^(-1/2),0,0,12.*pi.^(-1/2);0,0,0,0,0,0,( ...
15.*pi.^(-1)).^(1/2),0,3.*(15.*pi.^(-1)).^(1/2),0,0,0,0,3.*( ...
15.*pi.^(-1)).^(1/2),0;0,0,0,0,0,3.*(15.*pi.^(-1)).^(1/2),0, ...
(15.*pi.^(-1)).^(1/2),0,0,0,0,3.*(15.*pi.^(-1)).^(1/2),0,0; ...
12.*(5.*pi.^(-1)).^(1/2),0,(5.*pi.^(-1)).^(1/2),0,(-6).*(5.* ...
pi.^(-1)).^(1/2),0,0,0,0,(5.*pi.^(-1)).^(1/2),0,(-2).*(5.* ...
pi.^(-1)).^(1/2),0,0,(-6).*(5.*pi.^(-1)).^(1/2);0,3.*(15.* ...
pi.^(-1)).^(1/2),0,3.*(15.*pi.^(-1)).^(1/2),0,0,0,0,0,0,( ...
15.*pi.^(-1)).^(1/2),0,0,0,0;0,0,(15.*pi.^(-1)).^(1/2),0,6.* ...
(15.*pi.^(-1)).^(1/2),0,0,0,0,(-1).*(15.*pi.^(-1)).^(1/2),0, ...
0,0,0,(-6).*(15.*pi.^(-1)).^(1/2);0,0,0,0,0,0,0,0,(9/2).*( ...
35.*pi.^(-1)).^(1/2),0,0,0,0,(-9/2).*(35.*pi.^(-1)).^(1/2), ...
0;0,0,0,0,0,0,0,(9/2).*((35/2).*pi.^(-1)).^(1/2),0,0,0,0,( ...
-9/2).*((35/2).*pi.^(-1)).^(1/2),0,0;0,0,0,0,0,0,9.*(5.* ...
pi.^(-1)).^(1/2),0,(-9/2).*(5.*pi.^(-1)).^(1/2),0,0,0,0,( ...
-9/2).*(5.*pi.^(-1)).^(1/2),0;0,0,0,0,0,9.*(10.*pi.^(-1)).^( ...
1/2),0,(-9/2).*((5/2).*pi.^(-1)).^(1/2),0,0,0,0,(-27/2).*(( ...
5/2).*pi.^(-1)).^(1/2),0,0;36.*pi.^(-1/2),0,(-18).*pi.^( ...
-1/2),0,(27/2).*pi.^(-1/2),0,0,0,0,(-18).*pi.^(-1/2),0,(9/2) ...
.*pi.^(-1/2),0,0,(27/2).*pi.^(-1/2);0,9.*(10.*pi.^(-1)).^( ...
1/2),0,(-27/2).*((5/2).*pi.^(-1)).^(1/2),0,0,0,0,0,0,(-9/2) ...
.*((5/2).*pi.^(-1)).^(1/2),0,0,0,0;0,0,9.*(5.*pi.^(-1)).^( ...
1/2),0,(-9).*(5.*pi.^(-1)).^(1/2),0,0,0,0,(-9).*(5.*pi.^(-1) ...
).^(1/2),0,0,0,0,9.*(5.*pi.^(-1)).^(1/2);0,0,0,(9/2).*(( ...
35/2).*pi.^(-1)).^(1/2),0,0,0,0,0,0,(-9/2).*((35/2).*pi.^( ...
-1)).^(1/2),0,0,0,0;0,0,0,0,(9/2).*(35.*pi.^(-1)).^(1/2),0, ...
0,0,0,0,0,(-9/2).*(35.*pi.^(-1)).^(1/2),0,0,(9/2).*(35.* ...
pi.^(-1)).^(1/2)]' * q;

end

