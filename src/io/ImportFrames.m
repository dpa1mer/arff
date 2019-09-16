function frames = ImportFrames(filename)
    frames = dlmread(filename);
    frames = reshape(frames', 3, 3, []);
end

