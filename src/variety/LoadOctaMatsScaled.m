function octaMat = LoadOctaMatsScaled

octaMat = LoadOctaMats;
for j = 1:length(octaMat)
    octaMat{j} = blkdiag(sqrt(189)/4, eye(9)) ...
               * octaMat{j} ...
               * blkdiag(sqrt(189)/4, eye(9));
end

end