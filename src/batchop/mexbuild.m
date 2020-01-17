function mexbuild(pathToTbbInclude, pathToTbbLink)

%pathToTbbLink = 'C:\Users\pzpzp\Documents\MATLAB\tbb-tbb_2019\build\vs2013\x64\Release-MT';
%pathToTbbInclude = 'C:\Users\pzpzp\Documents\MATLAB\tbb-tbb_2019\include';

mex('batchop_cpu.cpp', '-O', '-R2018a', ['-I' pathToTbbInclude], ['-L' pathToTbbLink ],'-ltbb', '-lmwblas', '-lmwlapack');

% if gpuDeviceCount > 0
%     mexcuda -O -g batchop_gpu.cu -L/usr/local/cuda/lib64 -lcublas -lcusolver;
% end

end

