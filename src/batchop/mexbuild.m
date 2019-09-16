function mexbuild(pathToTbbInclude)

mex('batchop_cpu.cpp', '-O', '-R2018a', ['-I' pathToTbbInclude], '-ltbb', '-lmwblas', '-lmwlapack');
if gpuDeviceCount > 0
    mexcuda -O -g batchop_gpu.cu -L/usr/local/cuda/lib64 -lcublas -lcusolver;
end

end
