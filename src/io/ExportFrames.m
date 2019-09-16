function ExportFrames(filename, frames)
    frames = reshape(frames, 9, [])';
    writematrix(frames, filename, 'FileType', 'text', 'Delimiter', ' ');
end

