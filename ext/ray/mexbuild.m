function mexbuild(pathToEigen3, pathToTbbInclude)

mex('RayProjection.cpp', '-cxx', '-O', ['-I' pathToEigen3], ['-I' pathToTbbInclude], '-ltbb');
mex('RayInitHelper.cpp', '-cxx', '-O', 'ray_src/FF.cpp', 'ray_src/OpenNL_psm.cpp');

end